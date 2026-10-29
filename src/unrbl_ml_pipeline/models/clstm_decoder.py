"""Causal Conv1D-LSTM decoder for iEMG prosthesis control.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Architecture: causal Conv1D -> [BatchNorm] -> ReLU -> LSTM -> Dense(relu)
-> Dense(linear) -> [deadband] -> de-normalization.

Two build modes via the *stateful* flag:
  - stateful=False (default, training): LSTM starts from zero state each window.
  - stateful=True  (deployment): LSTM state (h, c) exposed as explicit I/O.
"""

name = "clstm_decoder"
description = "Causal Conv1D-LSTM decoder for iEMG prosthesis control"
paper = "Ferrisi (in preparation)"
model_type = "keras"

default_config = {
    "preprocessing_steps": [
        "load_kdf",
        "quality_filter",
        "split_by_detected_trial_deadzone",
        "normalize",
        "window",
    ],
    "model": {"window_size": 10, "prediction_lag_samples": 0},
    "stateful": False,
    "data": {"beta_threshold": 0.2},
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "epochs": 200,
    "batch_size": 128,
    "loss": {"type": "mse"},
    "split": {
        "test_size": 0.15, "val_size": 0.15, "random_state": 42,
        "trial_detection": {
            "min_trial_len": 10, "min_peak": 0.5, "merge_gap": 0,
            "active_eps": 1e-9, "non_trial_policy": "train",
        },
    },
    "early_stopping_patience": 18,
    "reduce_lr_patience": 8,
    "architecture": {
        "conv_filters": 64,
        "conv_kernel_size": 8,
        "use_batch_norm": True,
        "lstm_units": 128,
        "dense_units": 50,
        "dropout": 0.0,
        "use_deadband_activation_output": False,
    },
    "postprocessing": {
        "deadband": {"enabled": False, "threshold": 0.04},
        "latching_filter": {"enabled": False},
    },
}


def _deadband_activation(x, thresh=0.1):
    """Deadband activation: zero for |x| < thresh, passthrough otherwise."""
    import tensorflow as tf
    return tf.where(tf.abs(x) < thresh, tf.zeros_like(x), x)


def build(input_shape, num_classes, norm_stats=None, stateful=False, **config):
    """Return an uncompiled CLSTM decoder model.

    Parameters
    ----------
    input_shape : tuple
        ``(window_size, num_features)``.
    num_classes : int
        Number of kinematic DOFs to decode.
    norm_stats : object, optional
        Object with ``.feat_mean``, ``.feat_std``, ``.kin_mean``, ``.kin_std``.
    stateful : bool
        If True, LSTM hidden/cell state are explicit model inputs/outputs.
    **config
        Architecture overrides (merged with ``default_config["architecture"]``).
    """
    import keras
    from keras import layers

    arch = dict(default_config["architecture"])
    arch.update(config.get("architecture", {}))
    conv_filters = arch.get("conv_filters", 64)
    conv_kernel_size = arch.get("conv_kernel_size", 8)
    use_batch_norm = arch.get("use_batch_norm", True)
    lstm_units = arch.get("lstm_units", 128)
    dense_units = arch.get("dense_units", 50)
    dropout = arch.get("dropout", 0.0)
    use_deadband = arch.get("use_deadband_activation_output", False)

    window_size = input_shape[0] if len(input_shape) >= 2 else input_shape[-1]
    num_features = input_shape[-1]
    num_dofs = num_classes

    inputs = keras.Input(shape=(window_size, num_features), name="features")

    if stateful:
        h_in = keras.Input(shape=(lstm_units,), name="lstm_h_in")
        c_in = keras.Input(shape=(lstm_units,), name="lstm_c_in")

    x = inputs

    # Frozen normalization baked into the graph
    if norm_stats is not None:
        x = layers.Normalization(
            axis=-1,
            mean=norm_stats.feat_mean.tolist(),
            variance=(norm_stats.feat_std ** 2).tolist(),
            name="feature_normalization",
        )(x)

    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=conv_kernel_size,
        padding="causal",
        use_bias=not use_batch_norm,
        name="causal_conv1d",
    )(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name="conv_bn")(x)
    x = layers.ReLU(name="conv_relu")(x)

    # Recurrent core
    if stateful:
        x, h_out, c_out = layers.LSTM(
            lstm_units,
            unroll=True,
            dropout=0.0,
            recurrent_dropout=0.0,
            return_state=True,
            name="lstm",
        )(x, initial_state=[h_in, c_in])
    else:
        x = layers.LSTM(
            lstm_units,
            unroll=True,
            dropout=dropout,
            recurrent_dropout=0.0,
            name="lstm",
        )(x)

    # Readout head
    x = layers.Dense(dense_units, activation="relu", name="dense_relu")(x)
    y_norm = layers.Dense(
        num_dofs, activation="linear", name="dense_linear"
    )(x)

    # De-normalize and optional deadband
    prediction = y_norm
    if norm_stats is not None:
        prediction = layers.Rescaling(
            scale=norm_stats.kin_std.tolist(),
            offset=norm_stats.kin_mean.tolist(),
            name="denormalize_output",
        )(prediction)
    if use_deadband:
        prediction = keras.ops.where(
            keras.ops.abs(prediction) < 0.1,
            keras.ops.zeros_like(prediction), prediction,
        )

    if stateful:
        return keras.Model(
            inputs=[inputs, h_in, c_in],
            outputs=[prediction, h_out, c_out],
            name="clstm_decoder_stateful",
        )
    return keras.Model(
        inputs=inputs, outputs=prediction, name="clstm_decoder"
    )


def build_stateful_from_trained(
    trained_model, input_shape, num_classes, norm_stats=None, **config
):
    """Build a stateful model and copy weights from a trained non-stateful one."""
    stateful = build(
        input_shape,
        num_classes,
        norm_stats=norm_stats,
        stateful=True,
        **config,
    )
    for src_layer in trained_model.layers:
        try:
            dst_layer = stateful.get_layer(src_layer.name)
        except ValueError:
            continue
        src_weights = src_layer.get_weights()
        if src_weights:
            dst_layer.set_weights(src_weights)
    return stateful


def build_trajectory_wrapper(
    base_model, trajectory_length, learning_rate, loss_fn, metrics=None
):
    """TimeDistributed wrapper for trajectory losses (jerk penalty etc.).

    Input:  ``[batch, trajectory_length, window_size, num_features]``
    Output: ``[batch, trajectory_length, num_dofs]``
    """
    import keras
    from keras import layers

    trajectory_length = int(trajectory_length)
    if trajectory_length < 4:
        raise ValueError(
            "trajectory_length must be >= 4 for jerk computation, "
            f"got {trajectory_length}."
        )

    inp_shape = base_model.input_shape
    window_size = inp_shape[1]
    num_features = inp_shape[2]

    inp = keras.Input(
        shape=(trajectory_length, window_size, num_features),
        name="trajectory_feature_windows",
    )
    pred_seq = layers.TimeDistributed(
        base_model, name="shared_decoder_over_trajectory"
    )(inp)

    training_model = keras.Model(
        inp, pred_seq, name="trajectory_training_wrapper"
    )

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        float(learning_rate), decay_steps=100000, decay_rate=0.96,
        staircase=True,
    )
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=metrics or [],
    )
    return training_model


def build_stateful_trajectory_model(
    base_stateful, chunk_length, lstm_units, learning_rate, loss_fn,
    metrics=None
):
    """Stateful unrolled wrapper for truncated BPTT trajectory training.

    Input:  trajectory_features ``[batch, chunk_length, W, F]``,
            h0 ``[batch, lstm_units]``, c0 ``[batch, lstm_units]``
    Output: ``[batch, chunk_length, num_dofs]``
    """
    import keras
    import tensorflow as tf
    from keras import layers

    chunk_length = int(chunk_length)
    if chunk_length < 2:
        raise ValueError(
            f"chunk_length must be >= 2, got {chunk_length}."
        )

    inp_shape = base_stateful.input_shape[0]
    window_size = inp_shape[1]
    num_features = inp_shape[2]

    traj_inp = keras.Input(
        shape=(chunk_length, window_size, num_features),
        name="trajectory_features",
    )
    h0_inp = keras.Input(shape=(lstm_units,), name="h0")
    c0_inp = keras.Input(shape=(lstm_units,), name="c0")

    predictions = []
    h = h0_inp
    c = c0_inp
    for t in range(chunk_length):
        x_t = layers.Lambda(
            lambda x, idx=t: x[:, idx, :, :],
            name=f"slice_t{t}",
        )(traj_inp)
        pred_t, h, c = base_stateful([x_t, h, c])
        predictions.append(pred_t)

    pred_seq = layers.Lambda(
        lambda preds: tf.stack(preds, axis=1),
        name="stack_predictions",
    )(predictions)

    training_model = keras.Model(
        inputs=[traj_inp, h0_inp, c0_inp],
        outputs=pred_seq,
        name="stateful_trajectory_training_wrapper",
    )
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss=loss_fn,
        metrics=metrics or [],
    )
    return training_model


def stateful_predict(model, X, trial_id, lstm_units):
    """Run the stateful model in carry mode, resetting state at trial boundaries.

    Parameters
    ----------
    model : keras.Model
        Stateful model (3 inputs, 3 outputs).
    X : ndarray ``[N, window_size, num_features]``
    trial_id : ndarray ``[N]``
    lstm_units : int

    Returns
    -------
    ndarray ``[N, num_dofs]``
    """
    import numpy as np

    N = X.shape[0]
    h = np.zeros((1, lstm_units), dtype=np.float32)
    c = np.zeros((1, lstm_units), dtype=np.float32)
    preds = []
    prev_trial = None

    for i in range(N):
        if trial_id[i] != prev_trial:
            h = np.zeros((1, lstm_units), dtype=np.float32)
            c = np.zeros((1, lstm_units), dtype=np.float32)
            prev_trial = trial_id[i]

        pred, h, c = model.predict(
            {"features": X[i:i + 1], "lstm_h_in": h, "lstm_c_in": c},
            verbose=0,
        )
        preds.append(pred[0])

    return np.array(preds, dtype=np.float32)

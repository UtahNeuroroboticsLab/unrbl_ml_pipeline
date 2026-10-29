"""Caleb LSTM decoder: single-timestep LSTM from Vu et al. EMBC 2021.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Architecture from PMC12742979 (paper's third compared algorithm):
single timestep of raw EMG -> LSTM(128) -> Dense(1056, relu) x3 ->
Dense(num_dofs, linear).  No windowing, no conv front-end, no batch norm,
no output thresholding.
"""

name = "caleb_decoder"
description = "LSTM decoder from Vu et al. EMBC 2021 (PMC12742979)"
paper = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12742979/"
model_type = "keras"

default_config = {
    "preprocessing_steps": [
        "load_kdf",
        "quality_filter",
        "split_by_file",
        "normalize",
        "window",
    ],
    "window_size": 1,
    "beta_threshold": 0.45,
    "optimizer": "sgd",
    "learning_rate": 0.01,
    "epochs": 2000,
    "batch_size": 256,
    "loss": {"type": "mse"},
    "architecture": {
        "lstm_units": 128,
        "dense_units": 1056,
    },
}


def build(input_shape, num_classes, norm_stats=None, **config):
    """Return an uncompiled Caleb decoder model.

    Parameters
    ----------
    input_shape : tuple
        ``(1, num_features)`` -- single timestep.
    num_classes : int
        Number of kinematic DOFs to decode.
    norm_stats : object, optional
        Object with ``.feat_mean``, ``.feat_std``, ``.kin_mean``, ``.kin_std``
        arrays.  When provided, normalization is baked into the graph.
    **config
        Architecture overrides (merged with ``default_config["architecture"]``).
    """
    import keras
    from keras import layers

    arch = dict(default_config["architecture"])
    arch.update(config.get("architecture", {}))
    lstm_units = arch.get("lstm_units", 128)
    dense_units = arch.get("dense_units", 1056)

    num_features = input_shape[-1]
    num_dofs = num_classes

    inputs = keras.Input(shape=(1, num_features), name="features")
    x = inputs

    if norm_stats is not None:
        try:
            import tensorflow as tf
        except ImportError:
            tf = None
        if tf is not None:
            norm = layers.Normalization(axis=-1, name="feature_normalization")
            norm.mean = tf.constant(norm_stats.feat_mean, dtype=tf.float32)
            norm.variance = tf.constant(
                norm_stats.feat_std ** 2, dtype=tf.float32
            )
            norm.built = True
            x = norm(x)

    x = layers.LSTM(lstm_units, unroll=True, name="lstm")(x)

    x = layers.Dense(dense_units, activation="relu", name="fc1")(x)
    x = layers.Dense(dense_units, activation="relu", name="fc2")(x)
    x = layers.Dense(dense_units, activation="relu", name="fc3")(x)
    y_norm = layers.Dense(
        num_dofs, activation="linear", name="regression_output"
    )(x)

    if norm_stats is not None and tf is not None:
        kin_mean_t = tf.constant(norm_stats.kin_mean, dtype=tf.float32)
        kin_std_t = tf.constant(norm_stats.kin_std, dtype=tf.float32)
        outputs = layers.Lambda(
            lambda t: t * kin_std_t + kin_mean_t, name="denormalize_output"
        )(y_norm)
    else:
        outputs = y_norm

    return keras.Model(inputs=inputs, outputs=outputs, name="caleb_decoder")

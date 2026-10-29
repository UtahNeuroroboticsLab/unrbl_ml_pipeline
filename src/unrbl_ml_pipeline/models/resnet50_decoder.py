"""ResNet50 decoder for iEMG prosthesis control.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

The (window_size, num_features) EMG window is resized to (32, 32) and tiled
to 3 channels so ResNet50's fixed 2D-conv input contract is satisfied.
Random init (no ImageNet weights).
"""

name = "resnet50_decoder"
description = "ResNet50 decoder for iEMG prosthesis control"
model_type = "keras"

default_config = {
    "preprocessing_steps": [
        "load_kdf",
        "quality_filter",
        "split_by_file",
        "normalize",
        "window",
    ],
    "model": {"window_size": 30},
    "data": {"beta_threshold": 0.2},
    "optimizer": "sgd",
    "learning_rate": 0.0001,
    "epochs": 2000,
    "batch_size": 256,
    "loss": {"type": "recency_weighted_mse"},
    "architecture": {
        "image_size": 32,
    },
}


def build(input_shape, num_classes, norm_stats=None, **config):
    """Return an uncompiled ResNet50 decoder model.

    Parameters
    ----------
    input_shape : tuple
        ``(window_size, num_features)``.
    num_classes : int
        Number of kinematic DOFs to decode.
    norm_stats : object, optional
        Object with ``.feat_mean``, ``.feat_std``, ``.kin_mean``, ``.kin_std``.
    **config
        Architecture overrides.
    """
    import keras
    from keras import layers

    arch = dict(default_config["architecture"])
    arch.update(config.get("architecture", {}))
    image_size = arch.get("image_size", 32)

    window_size = input_shape[0] if len(input_shape) >= 2 else input_shape[-1]
    num_features = input_shape[-1]
    num_dofs = num_classes

    inputs = keras.Input(shape=(window_size, num_features), name="features")
    x = inputs

    if norm_stats is not None:
        x = layers.Normalization(
            axis=-1,
            mean=norm_stats.feat_mean.tolist(),
            variance=(norm_stats.feat_std ** 2).tolist(),
            name="feature_normalization",
        )(x)

    x = layers.Reshape(
        (window_size, num_features, 1), name="add_channel_dim"
    )(x)
    x = layers.Resizing(image_size, image_size, name="resize_to_image")(x)
    x = layers.Concatenate(axis=-1, name="tile_to_3ch")([x, x, x])

    backbone = keras.applications.ResNet50(
        include_top=False,
        weights=None,
        pooling="avg",
        input_shape=(image_size, image_size, 3),
    )
    x = backbone(x)

    y_norm = layers.Dense(
        num_dofs, activation="linear", name="regression_output"
    )(x)

    outputs = y_norm
    if norm_stats is not None:
        outputs = layers.Rescaling(
            scale=norm_stats.kin_std.tolist(),
            offset=norm_stats.kin_mean.tolist(),
            name="denormalize_output",
        )(y_norm)

    return keras.Model(
        inputs=inputs, outputs=outputs, name="resnet50_decoder"
    )

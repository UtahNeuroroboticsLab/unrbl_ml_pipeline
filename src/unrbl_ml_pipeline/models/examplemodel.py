"""Example model: a simple feedforward network.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Copy this file, rename it, and replace the architecture in build().

Required exports
    build(input_shape, num_classes) -> keras.Model (uncompiled)

Optional exports
    name        : str  -- canonical lowercase name (defaults to filename)
    description : str  -- one-line summary
    paper       : str  -- citation or URL
"""

name = "examplemodel"
description = "Simple dense network for testing and as a model-file template"


def build(input_shape, num_classes):
    """Return an uncompiled feedforward classifier.

    Parameters
    ----------
    input_shape : tuple
        Shape of one sample, e.g. (128, 8) for 128 timesteps x 8 channels.
    num_classes : int
        Number of output units (classes for classification, outputs for regression).
    """
    import keras
    from keras import layers

    # MODEL DEFINTION GOES AROUND HERE AND IS RETURNED YOU CAN MAKE MORE...
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes),
    ])

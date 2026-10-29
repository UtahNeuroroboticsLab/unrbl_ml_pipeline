"""Core training entry point.

Author: Leonardo M. Ferrisi
Date: 2026-10-29
"""

import keras
import numpy as np


def train(
    model,
    x,
    y,
    task="classification",
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    optimizer="adam",
    loss=None,
    metrics=None,
    callbacks=None,
    save_path=None,
    **fit_kwargs,
):
    """Build (if needed), compile, and fit a model on the given data.

    Parameters
    ----------
    model : module | str | keras.Model
        A model module from ``unrbl_ml_pipeline.models`` (has a ``build``
        attribute), a registered model name string, or an already-built
        ``keras.Model``.
    x : numpy array
        Training inputs.  Shape ``(samples, ...)``.
    y : numpy array
        Labels.  For classification: 1-D integer labels or 2-D one-hot.
        For regression: 1-D or 2-D continuous targets.
    task : str
        ``"classification"`` (default) or ``"regression"``.
    epochs, batch_size, validation_split, optimizer, loss, metrics, callbacks
        Forwarded to ``model.compile`` / ``model.fit`` with task-appropriate
        defaults when not provided.
    save_path : str, optional
        If given, save the trained model to this path (as ``.keras``).
    **fit_kwargs
        Extra keyword arguments passed through to ``model.fit()``.

    Returns
    -------
    (keras.Model, keras.callbacks.History)
    """
    if task not in ("classification", "regression"):
        raise ValueError(
            f"task must be 'classification' or 'regression', got {task!r}"
        )

    keras_model = _resolve_model(model, x, y, task)
    _compile_model(keras_model, task, optimizer, loss, metrics)

    y_fit = _prepare_labels(y, task)

    history = keras_model.fit(
        x,
        y_fit,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        **fit_kwargs,
    )

    if save_path is not None:
        keras_model.save(save_path)

    return keras_model, history


def _resolve_model(model, x, y, task):
    """Turn *model* into a built ``keras.Model``."""
    if isinstance(model, keras.Model):
        return model

    if isinstance(model, str):
        from unrbl_ml_pipeline import models
        model = models.get(model)

    if not hasattr(model, "build"):
        raise TypeError(
            "Expected a model module with a build() function, a model name "
            f"string, or a keras.Model instance. Got {type(model).__name__}"
        )

    input_shape = x.shape[1:]
    num_classes = _infer_num_outputs(y, task)
    return model.build(input_shape, num_classes)


def _infer_num_outputs(y, task):
    """Infer the output dimension from *y* and *task*."""
    if task == "classification":
        if y.ndim == 1:
            return int(np.max(y)) + 1
        return y.shape[-1]
    # regression
    if y.ndim == 1:
        return 1
    return y.shape[-1]


def _prepare_labels(y, task):
    """One-hot encode integer class labels when needed."""
    if task == "classification" and y.ndim == 1:
        num_classes = int(np.max(y)) + 1
        return keras.utils.to_categorical(y, num_classes)
    return y


def _compile_model(model, task, optimizer, loss, metrics):
    """Compile *model* with task-appropriate defaults."""
    if task == "classification":
        if loss is None:
            loss = "categorical_crossentropy"
        if metrics is None:
            metrics = ["accuracy"]
    else:
        if loss is None:
            loss = "mse"
        if metrics is None:
            metrics = ["mae"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

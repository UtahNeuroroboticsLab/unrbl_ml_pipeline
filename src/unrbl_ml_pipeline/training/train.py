"""Core training entry point.

Author: Leonardo M. Ferrisi
Date: 2026-10-29
"""
from __future__ import annotations

import keras
import numpy as np

from .config import resolve_config


def train(
    model,
    x=None,
    y=None,
    task="classification",
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    optimizer="adam",
    loss=None,
    metrics=None,
    callbacks=None,
    save_path=None,
    config=None,
    config_path=None,
    data_dir=None,
    **fit_kwargs,
):
    """Build (if needed), compile, and fit a model on the given data.

    Three routing modes:

    1. **Simple** (x/y provided): legacy path — build, compile, fit.
    2. **Preprocessing** (data_dir or config with preprocessing_steps, model_type="keras"):
       run preprocessing pipeline, build model with norm_stats, compile with
       custom loss, fit, postprocess.
    3. **Estimator** (model_type="estimator"): same preprocessing, then
       ``model.fit(X, Y)`` and ``model.predict(X)`` generically.

    Parameters
    ----------
    model : module | str | keras.Model
        A model module, registered model name, or a keras.Model.
    x, y : numpy array, optional
        Training inputs/targets.  When omitted, *data_dir* + preprocessing
        steps must provide data.
    config : dict | str, optional
        User overrides dict, or path to a .jsonc config file.
    config_path : str, optional
        Path to a .jsonc config file (alternative to passing config as str).
    data_dir : str, optional
        Base directory for .kdf file discovery (injected into config).
    save_path : str, optional
        If given, save the trained model to this path.
    **fit_kwargs
        Extra keyword arguments passed through to ``model.fit()``.

    Returns
    -------
    (model, history_or_results)
    """
    mod = _resolve_module(model)

    if isinstance(config, str):
        config_path = config
        config = None

    merged = resolve_config(mod, user_config=config, config_path=config_path)

    if data_dir is not None:
        merged.setdefault("data", {})["base_dir"] = data_dir

    model_type = getattr(mod, "model_type", "keras")
    has_pipeline = "preprocessing_steps" in merged and (data_dir or x is None)

    if x is not None and not has_pipeline:
        return _train_simple(
            mod, x, y, task=task, epochs=epochs, batch_size=batch_size,
            validation_split=validation_split, optimizer=optimizer,
            loss=loss, metrics=metrics, callbacks=callbacks,
            save_path=save_path, **fit_kwargs,
        )

    if model_type == "estimator":
        return _train_estimator(mod, merged, save_path=save_path)

    return _train_with_preprocessing(
        mod, merged, save_path=save_path,
        extra_callbacks=callbacks, **fit_kwargs,
    )


def _resolve_module(model):
    """Return the model module (not a built keras.Model)."""
    if isinstance(model, str):
        from unrbl_ml_pipeline import models
        return models.get(model)
    if hasattr(model, "build"):
        return model
    if isinstance(model, keras.Model):
        return model
    raise TypeError(
        "Expected a model module with a build() function, a model name "
        f"string, or a keras.Model instance. Got {type(model).__name__}"
    )


def _train_simple(
    model, x, y, task="classification", epochs=50, batch_size=32,
    validation_split=0.2, optimizer="adam", loss=None, metrics=None,
    callbacks=None, save_path=None, **fit_kwargs,
):
    """Legacy path: x/y provided, simple build+compile+fit."""
    if task not in ("classification", "regression"):
        raise ValueError(
            f"task must be 'classification' or 'regression', got {task!r}"
        )

    keras_model = _build_from_xy(model, x, y, task)
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


def _train_with_preprocessing(mod, config, save_path=None,
                              extra_callbacks=None, **fit_kwargs):
    """Preprocessing pipeline path for keras models."""
    from .losses import (
        get_loss,
        is_trajectory_loss,
        make_predicted_trajectory_loss_fast,
    )
    from .postprocessing import apply_postprocessing
    from .preprocessing import run_pipeline

    steps = config["preprocessing_steps"]
    state = run_pipeline(steps, config)

    norm_stats = state.norm_stats
    build_kwargs = dict(config)
    build_kwargs.pop("preprocessing_steps", None)

    if state.X_train is not None:
        input_shape = state.X_train.shape[1:]
        num_classes = state.Y_train.shape[-1]
    else:
        raise ValueError("Preprocessing pipeline produced no training data.")

    keras_model = mod.build(
        input_shape, num_classes, norm_stats=norm_stats, **build_kwargs,
    )

    loss_cfg = config.get("loss", {"type": "mse"})
    loss_type = loss_cfg.get("type", "mse")

    if is_trajectory_loss(loss_type):
        loss_fn = make_predicted_trajectory_loss_fast(**{
            k: v for k, v in loss_cfg.items() if k != "type"
        })
    else:
        loss_fn, _ = get_loss(loss_cfg)

    opt_name = config.get("optimizer", "adam")
    lr = config.get("learning_rate", 0.001)
    optimizer = _make_optimizer(opt_name, lr)

    keras_model.compile(optimizer=optimizer, loss=loss_fn, metrics=["mae"])

    cbs = list(extra_callbacks or [])
    if config.get("early_stopping_patience"):
        cbs.append(keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
        ))
    if config.get("reduce_lr_patience"):
        cbs.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=config["reduce_lr_patience"],
            factor=0.5,
            min_lr=1e-7,
        ))

    train_data = state.train_sequence if state.train_sequence is not None else state.X_train
    val_data = None
    if state.val_sequence is not None:
        val_data = state.val_sequence
    elif state.X_val is not None:
        val_data = (state.X_val, state.Y_val)

    if state.train_sequence is not None:
        history = keras_model.fit(
            train_data,
            epochs=config.get("epochs", 200),
            validation_data=val_data,
            callbacks=cbs or None,
            **fit_kwargs,
        )
    else:
        history = keras_model.fit(
            state.X_train,
            state.Y_train,
            epochs=config.get("epochs", 200),
            batch_size=config.get("batch_size", 128),
            validation_data=val_data,
            callbacks=cbs or None,
            **fit_kwargs,
        )

    if save_path is not None:
        keras_model.save(save_path)

    results = {"history": history, "state": state}

    if state.X_test is not None and state.X_test.shape[0] > 0:
        Y_pred = keras_model.predict(state.X_test)
        pp_cfg = config.get("postprocessing")
        if pp_cfg and state.trial_id_test is not None:
            Y_pred = apply_postprocessing(Y_pred, state.trial_id_test, pp_cfg)
        results["Y_pred"] = Y_pred
        results["Y_test"] = state.Y_test
        results["test_loss"] = float(keras_model.evaluate(
            state.X_test, state.Y_test, verbose=0,
        )[0])

    return keras_model, results


def _train_estimator(mod, config, save_path=None):
    """Estimator path for non-Keras models (KF, sklearn, etc.)."""
    from .postprocessing import apply_postprocessing
    from .preprocessing import run_pipeline

    steps = config.get("preprocessing_steps", [])
    state = run_pipeline(steps, config) if steps else None

    estimator_cls = mod.build(**config)

    if state is not None and state.train_files:
        if hasattr(estimator_cls, "fit_multi_file"):
            fitted = estimator_cls.fit_multi_file(state.train_files)
        else:
            train_feat = np.concatenate(
                [f.features for f in state.train_files], axis=0,
            )
            train_kin = np.concatenate(
                [f.kinematics for f in state.train_files], axis=0,
            )
            fitted = estimator_cls()
            fitted.fit(train_feat, train_kin)
    elif state is not None and state.X_train is not None:
        fitted = estimator_cls()
        fitted.fit(state.X_train, state.Y_train)
    else:
        raise ValueError("No training data available for estimator.")

    if save_path is not None:
        fitted.save(save_path)

    results = {"state": state}

    if state is not None and state.test_files:
        test_feat = np.concatenate(
            [f.features for f in state.test_files], axis=0,
        )
        test_kin = np.concatenate(
            [f.kinematics for f in state.test_files], axis=0,
        )
        Y_pred = fitted.predict(test_feat)
        pp_cfg = config.get("postprocessing")
        if pp_cfg:
            trial_id = np.concatenate([
                np.full(f.features.shape[0], f.trial_id)
                for f in state.test_files
            ])
            Y_pred = apply_postprocessing(Y_pred, trial_id, pp_cfg)
        results["Y_pred"] = Y_pred
        results["Y_test"] = test_kin
        results["test_mse"] = float(np.mean((Y_pred - test_kin) ** 2))

    return fitted, results


def _make_optimizer(name, lr):
    """Build a Keras optimizer from name + learning rate."""
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    return keras.optimizers.get({"class_name": name, "config": {"learning_rate": lr}})


def _build_from_xy(model, x, y, task):
    """Turn *model* into a built ``keras.Model`` using x/y shapes."""
    if isinstance(model, keras.Model):
        return model
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

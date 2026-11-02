"""Mamba training with explicit stream state, TBPTT and epoch checkpoints."""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .mamba_data import batches, predict, validate_metadata
from .mamba_objectives import (
    calibrate,
    jacobian_probe,
    mse,
    predict_chunk,
    temporal_terms,
)


def fingerprint(state, config):
    recipe = {
        k: v
        for k, v in config.items()
        if k not in ("epochs", "resume_from", "output_dir")
    }
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode())
    for split in ("train", "val", "test"):
        for name in ("X_", "Y_"):
            a = getattr(state, name + split)
            digest.update(str(a.shape).encode())
            digest.update(memoryview(np.ascontiguousarray(a)).cast("B"))
        for r in state.provenance[split]:
            digest.update((r["path"] + r["segment_id"]).encode())
            for key in ("target_indices", "input_end_indices", "target_times"):
                digest.update(np.asarray(r[key]).tobytes())
    return digest.hexdigest()


def save_checkpoint(root, model, optimizer, best_weights, metadata):
    """Commit immutable epoch contents, then atomically publish the pointer."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"epoch_{metadata['epoch']:05d}_{uuid.uuid4().hex[:8]}"
    directory.mkdir()
    arrays = {f"model_{i}": v for i, v in enumerate(model.get_weights())}
    arrays.update({f"best_{i}": v for i, v in enumerate(best_weights)})
    arrays.update(
        {f"optimizer_{i}": v.numpy() for i, v in enumerate(optimizer.variables)}
    )
    arrays["dropout_rng"] = model.rng.state.numpy()
    np.savez(directory / "state.npz", **arrays)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)
    )
    tmp = root / ("latest." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps({"directory": directory.name}))
    os.replace(tmp, root / "latest.json")
    return root


def restore_checkpoint(path, model, optimizer, expected):
    path = Path(path)
    if (path / "checkpoints").is_dir():
        path = path / "checkpoints"
    if (path / "latest.json").is_file():
        path = path / json.loads((path / "latest.json").read_text())["directory"]
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["fingerprint"] != expected:
        raise ValueError(
            "Resume data/config fingerprint mismatch: architecture, mode, preprocessing or recipe changed"
        )
    with np.load(path / "state.npz", allow_pickle=False) as values:
        model.set_weights(
            [values[f"model_{i}"] for i in range(len(model.get_weights()))]
        )
        for i, v in enumerate(optimizer.variables):
            v.assign(values[f"optimizer_{i}"])
        model.rng.state.assign(values["dropout_rng"])
        best = [values[f"best_{i}"].copy() for i in range(len(model.get_weights()))]
    return metadata, best


def train_mamba(
    mod, config, progress, save_path=None, callbacks=None, metrics=None, **kwargs
):
    from .postprocessing import apply_postprocessing
    from .preprocessing import run_pipeline

    verbose = kwargs.pop("verbose", 1)
    shuffle = kwargs.pop("shuffle", True)
    if kwargs or metrics is not None:
        raise ValueError(
            "Mamba trainer accepts verbose/shuffle and callbacks; arbitrary fit kwargs or custom metrics are unsupported"
        )
    if shuffle not in (True, False):
        raise ValueError(
            "Mamba shuffle must be boolean (segment order only in carry mode)"
        )
    for key in ("epochs", "batch_size", "streams", "tbptt_steps", "min_epochs"):
        if int(config.get(key, 1)) < 1:
            raise ValueError(f"{key} must be positive")
    loss_cfg = config.get("loss", {"type": "mse"})
    objective = loss_cfg["type"]
    pinn, jac = "pinn" in objective, "jacobian" in objective
    warmup = int(loss_cfg.get("warmup_epochs", 20))
    if warmup < 0 or ((pinn or jac) and config["epochs"] <= warmup):
        raise ValueError(
            "Regularized training requires epochs > nonnegative warmup_epochs"
        )
    if config.get("optimizer", "adamw").lower() != "adamw":
        raise ValueError("The Mamba reference trainer supports optimizer=adamw")
    keras.utils.set_random_seed(config.get("seed", 0))
    state = run_pipeline(config["preprocessing_steps"], config, progress=progress)
    validate_metadata(state, config)
    selected = state.fitted_preprocessing.get("feature_selection", {}).get("indices")
    progress.stage(f"Building {mod.name} model")
    model = mod.build(
        state.X_train.shape[1:],
        state.Y_train.shape[-1],
        norm_stats=state.norm_stats,
        selected_indices=selected,
        **config,
    )
    if verbose:
        progress.message(
            "Summary shapes use an independent-window build pass (batch=1). "
            "MambaBlock outputs: (sequence, (convolution state, SSM state)). "
            "Carry execution uses variable batch/chunk lengths."
        )
        model.summary()
    from .mamba_layers import ReferenceAdamW

    optimizer = ReferenceAdamW(
        learning_rate=config.get("learning_rate", 0.0005),
        weight_decay=config.get("weight_decay", 0.0001),
        global_clipnorm=config.get("gradient_clip", 1.0),
    )
    optimizer.exclude_from_weight_decay(var_list=[model.mamba.A_log, model.mamba.D])
    optimizer.build(model.trainable_variables)
    model.compile(optimizer=optimizer, loss="mse")
    rng = np.random.default_rng(config.get("seed", 0))
    # Float64 accumulators avoid cancellation for high-amplitude raw EMG.
    scale = state.X_train.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale[scale <= 1e-8] = 0
    scale = tf.constant(scale)
    identity = fingerprint(state, config)
    best, best_epoch, bad, first = float("inf"), 0, 0, 1
    best_weights, calibration = model.get_weights(), None
    history = keras.callbacks.History()
    history.history = {}
    resume = config.get("resume_from")
    if resume:
        saved, best_weights = restore_checkpoint(resume, model, optimizer, identity)
        best, best_epoch, bad, first = (
            saved["best"],
            saved["best_epoch"],
            saved["bad"],
            saved["epoch"] + 1,
        )
        calibration, history.history = saved["calibration"], saved["history"]
        rng.bit_generator.state = saved["numpy_rng"]
        if config["epochs"] < saved["epoch"]:
            raise ValueError("epochs cannot precede the resumed checkpoint epoch")
    history.epoch = list(range(first - 1))
    cb = keras.callbacks.CallbackList(
        list(callbacks or []), model=model, epochs=config["epochs"], verbose=verbose
    )
    model.stop_training = False
    checkpoint_root = (
        Path(save_path).parent / "checkpoints" if save_path is not None else None
    )
    if resume and checkpoint_root is None:
        raise ValueError(
            "Resumed training requires save_path for its checkpoint output"
        )

    @tf.function(reduce_retracing=True)
    def update(
        x,
        y,
        mask,
        edges,
        runtime,
        px,
        py,
        pmask,
        pedges,
        weights,
        jw,
        seed,
        position,
        active,
        reset,
    ):
        with tf.GradientTape() as tape:
            if model.state_mode == "carry" and reset:
                runtime = model.prefill(x[:, 0, :-1], training=True)
            pred, new_state = predict_chunk(model, x, runtime, training=True)
            base = mse(pred, y, mask)
            terms, jterm = tf.zeros(3), tf.constant(0.0, tf.float32)
            if active:
                if pinn:
                    if model.state_mode == "reset":
                        pp, _ = predict_chunk(model, px, (), training=True)
                        terms = temporal_terms(pp, py, pedges)
                    else:
                        terms = temporal_terms(pred, y, edges)
                if jac:
                    jterm = jacobian_probe(
                        model, x, runtime, mask, scale, seed, position, training=True
                    )
            total = base + tf.reduce_sum(weights * terms) + jw * jterm
            tf.debugging.assert_all_finite(total, "Nonfinite training objective")
        grads = tape.gradient(total, model.trainable_variables)
        for gradient in grads:
            if gradient is not None:
                tf.debugging.assert_all_finite(gradient, "Nonfinite training gradient")
        optimizer.apply_gradients(
            (g, v) for g, v in zip(grads, model.trainable_variables) if g is not None
        )
        return base, total, terms, jterm, tuple(tf.stop_gradient(v) for v in new_state)

    def calibration_batches():
        return batches(
            state, "train", config.get("streams", 4), config.get("tbptt_steps", 16)
        )

    def reset_batches():
        order = (
            rng.permutation(len(state.X_train))
            if shuffle
            else np.arange(len(state.X_train))
        )
        size = config.get("batch_size", 64)
        for a in range(0, len(order), size):
            ids = order[a : a + size]
            yield (
                state.X_train[ids, None],
                state.Y_train[ids, None],
                np.ones((len(ids), 1), bool),
                np.zeros((len(ids), 0), bool),
                ids[:, None],
                True,
            )

    progress.stage("Training")
    if verbose:
        progress.message(
            "objective = MSE + weighted temporal penalties + weighted Jacobian penalty"
        )
        progress.message(
            f"Selected loss: {objective}; "
            f"temporal penalties: {'enabled' if pinn else 'disabled'}; "
            f"Jacobian penalty: {'enabled' if jac else 'disabled'}."
        )
        if pinn or jac:
            progress.message(
                f"Configured warmup: {warmup} epochs; penalties contribute only "
                "after calibration. During warmup, objective = MSE."
            )
        else:
            progress.message("No penalties enabled: objective = MSE.")
        progress.message(
            "loss/mse and val_loss/val_mse are plain MSE; objective includes active "
            "regularizers. Progress counts valid training samples, excluding padding."
        )
    cb.on_train_begin()
    for epoch in range(first, config["epochs"] + 1):
        if epoch > config.get("min_epochs", 145) and bad >= config.get(
            "early_stopping_patience", 45
        ):
            break
        if (pinn or jac) and calibration is None and epoch > warmup:
            calibration = calibrate(
                model, calibration_batches(), scale, loss_cfg, pinn, jac
            )
        active = calibration is not None
        weights = tf.constant(
            calibration["temporal_weights"] if active else [0.0, 0.0, 0.0], tf.float32
        )
        jw = tf.constant(calibration["jacobian_weight"] if active else 0.0, tf.float32)
        loader = (
            batches(
                state,
                "train",
                config.get("streams", 4),
                config.get("tbptt_steps", 16),
                rng if shuffle else None,
            )
            if model.state_mode == "carry"
            else reset_batches()
        )
        runtime, total, count = (), np.zeros(6), 0
        cb.on_epoch_begin(epoch - 1)
        epoch_start = time.perf_counter()
        if verbose:
            progress.message(f"Epoch {epoch}/{config['epochs']}")
        display_metrics = (
            "loss",
            "mse",
            "rmse",
            "objective",
            "val_loss",
            "val_mse",
            "val_rmse",
            "learning_rate",
            "epoch_seconds",
        )
        bar = keras.utils.Progbar(
            len(state.Y_train),
            verbose=verbose,
            unit_name="sample",
            stateful_metrics=display_metrics,
        )
        for number, (x, y, mask, edges, _, reset) in enumerate(loader):
            if reset and model.state_mode == "carry":
                runtime = model.initial_state(len(x))
            # Independent contiguous reset-mode probes, never differences across shuffled windows.
            px, py, pm, pe = x, y, mask, edges
            if active and pinn and model.state_mode == "reset":
                record = state.provenance["train"][
                    int(rng.integers(len(state.provenance["train"])))
                ]
                a = int(
                    rng.integers(
                        record["start"], max(record["start"] + 1, record["end"] - 15)
                    )
                )
                b = min(record["end"], a + 16)
                px, py = state.X_train[None, a:b], state.Y_train[None, a:b]
                pm = np.ones((1, b - a), bool)
                times = np.asarray(record["target_times"], dtype=np.float64)[
                    a - record["start"] : b - record["start"]
                ]
                pe = (
                    np.abs(np.diff(times) - record["nominal_interval"])
                    <= 0.1 * record["nominal_interval"]
                )[None]
            position = (
                int(rng.choice(np.flatnonzero(mask.any(0)))) if active and jac else 0
            )
            seed = tf.constant(rng.integers(0, 2**30, 2), tf.int32)
            cb.on_train_batch_begin(number)
            base, loss, terms, jterm, runtime = update(
                *(tf.convert_to_tensor(v) for v in (x, y, mask, edges)),
                runtime,
                *(tf.convert_to_tensor(v) for v in (px, py, pm, pe)),
                weights,
                jw,
                seed,
                tf.constant(position, tf.int32),
                active,
                tf.constant(reset),
            )
            n = int(mask.sum())
            total += np.r_[float(base), float(loss), terms.numpy(), float(jterm)] * n
            count += n
            cb.on_train_batch_end(
                number,
                {
                    "loss": float(base),
                    "mse": float(base),
                    "rmse": float(np.sqrt(base)),
                    "objective": float(loss),
                },
            )
            running_mse = total[0] / count
            bar.update(
                count,
                [
                    ("loss", running_mse),
                    ("mse", running_mse),
                    ("rmse", float(np.sqrt(running_mse))),
                    ("objective", total[1] / count),
                ],
                finalize=False,
            )
        if count != len(state.Y_train):
            raise ValueError("An epoch must train on every valid sample exactly once")
        val = predict(model, state, "val", config)
        val_loss = float(np.mean((val.astype(np.float64) - state.Y_val) ** 2))
        score = float(np.sqrt(val_loss))
        if not np.isfinite(score):
            raise ValueError("Nonfinite validation RMSE")
        if score < best:
            best, best_epoch, best_weights, bad = score, epoch, model.get_weights(), 0
        else:
            bad += 1
        values = total / count
        logs = dict(
            zip(("loss", "objective", "d1", "d2", "d3", "jacobian"), values.tolist())
        )
        logs.update(
            mse=values[0],
            rmse=float(np.sqrt(values[0])),
            val_loss=val_loss,
            val_mse=val_loss,
            val_rmse=score,
            learning_rate=float(optimizer.learning_rate.numpy()),
            epoch_seconds=time.perf_counter() - epoch_start,
            regularizers_active=int(active),
        )
        for key, value in logs.items():
            # Older checkpoints lack the new display fields. Preserve epoch indexing.
            history.history.setdefault(key, [None] * (epoch - 1)).append(value)
        history.epoch.append(epoch - 1)
        bar.update(count, [(key, logs[key]) for key in display_metrics], finalize=True)
        cb.on_epoch_end(epoch - 1, logs)
        # Reference calibration occurs at the end of the last warmup epoch.
        if (pinn or jac) and calibration is None and epoch == warmup:
            calibration = calibrate(
                model, calibration_batches(), scale, loss_cfg, pinn, jac
            )
        if checkpoint_root is not None:
            save_checkpoint(
                checkpoint_root,
                model,
                optimizer,
                best_weights,
                {
                    "epoch": epoch,
                    "best": best,
                    "best_epoch": best_epoch,
                    "bad": bad,
                    "history": history.history,
                    "calibration": calibration,
                    "fingerprint": identity,
                    "numpy_rng": rng.bit_generator.state,
                },
            )
        if model.stop_training:
            break
    cb.on_train_end()
    model.set_weights(best_weights)
    if save_path is not None:
        progress.stage("Saving model")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
    progress.stage("Evaluating on held-out test split")
    raw = predict(model, state, "test", config)
    # Unique segment IDs prevent filtering across equal trial labels in different files.
    trial = np.concatenate(
        [
            np.full(r["end"] - r["start"], r["segment_id"])
            for r in state.provenance["test"]
        ]
    )
    pred = apply_postprocessing(raw.copy(), trial, config.get("postprocessing", {}))
    history.set_model(model)
    result = {
        "history": history,
        "state": state,
        "config": config,
        "Y_pred_raw": raw,
        "Y_pred": pred,
        "Y_test": state.Y_test,
        "trial_id_test": trial,
        "test_loss": float(np.mean((raw.astype(np.float64) - state.Y_test) ** 2)),
        "calibration": calibration,
        "fitted_preprocessing": state.fitted_preprocessing,
        "best_epoch": best_epoch,
        "best_checkpoint_full_strength": not (pinn or jac) or best_epoch > warmup,
        "resume_from": resume,
        "checkpoint_dir": checkpoint_root,
    }
    return model, result

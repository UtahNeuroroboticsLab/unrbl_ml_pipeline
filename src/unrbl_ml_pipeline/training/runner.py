"""Training runs with reproducible records and headless diagnostic plots."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .config import resolve_config
from .train import _resolve_module, train


def create_run_directory(output_dir, model_name):
    """Create a new directory without overwriting any prior run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(output_dir).resolve() / f"{model_name}_{stamp}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path, value):
    def encode(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Cannot record {type(obj).__name__} in JSON")

    path.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + "\n",
                    encoding="utf-8")


def regression_metrics(target, prediction, names=None):
    """Aggregate and per-DOF errors in original target units."""
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or target.ndim != 2 or not len(target):
        raise ValueError("Metrics require nonempty matching [samples, DOFs] arrays")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Cannot report metrics for non-finite targets or predictions")
    error = prediction.astype(np.float64) - target
    mse = np.mean(error ** 2, axis=0)
    mae = np.mean(np.abs(error), axis=0)
    names = names or [f"dof_{i}" for i in range(target.shape[1])]
    if len(names) != target.shape[1]:
        raise ValueError("DOF names must match the target dimension")
    return {
        "samples": len(target),
        "aggregate": {"mse": float(mse.mean()), "rmse": float(np.sqrt(mse.mean())),
                      "mae": float(mae.mean())},
        "per_dof": [{"name": name, "mse": float(mse[i]), "rmse": float(np.sqrt(mse[i])),
                     "mae": float(mae[i])} for i, name in enumerate(names)],
    }


def _plots(run_dir, results, names):
    # FigureCanvasAgg avoids display/backend changes in callers' applications.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    history = results.get("history")
    if history is not None and "loss" in history.history:
        fig = Figure(figsize=(8, 4), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        for key, style in (("loss", "-"), ("val_loss", "--")):
            values = history.history.get(key)
            if values is not None:
                ax.plot(np.arange(1, len(values) + 1), values, style, color="black",
                        marker="o" if len(values) == 1 else None, label=key)
        ax.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
        ax.legend(frameon=False)
        fig.savefig(run_dir / "loss.png", dpi=150)
        fig.clear()

    truth, pred = results["Y_test"], results["Y_pred"]
    ids = results["trial_id_test"]
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    stride = max(1, len(ids) // 10000)
    # NaNs break lines between segments, even after plotting subsampling.
    x, indices = [], []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        selected = np.unique(np.r_[np.arange(start, end, stride), end - 1])
        x.extend(selected.tolist() + [np.nan])
        indices.extend(selected.tolist() + [-1])
    indices = np.asarray(indices)
    fig = Figure(figsize=(12, 2.5 * truth.shape[1]), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(truth.shape[1], 1, squeeze=False)[:, 0]
    for d, ax in enumerate(axes):
        for values, label, style, color in ((truth, "Target", "-", "black"),
                                            (pred, "Prediction", "--", "0.4")):
            y = values[np.maximum(indices, 0), d].astype(float)
            y[indices < 0] = np.nan
            ax.plot(x, y, style, color=color, linewidth=0.8, label=label)
        ax.set_ylabel(names[d])
    axes[0].legend(frameon=False)
    axes[0].set_title("Held-out predictions (gaps mark trial/recording boundaries)")
    axes[-1].set_xlabel("Sample index in concatenated test segments")
    fig.savefig(run_dir / "predictions.png", dpi=150)
    fig.clear()


def run_training(model="clstm_decoder", *, config=None, config_path=None,
                 data_dir=None, output_dir=None, **train_kwargs):
    """Call train(), save a unique run, and return (model, results).

    Configuration order: model defaults < config_path < config < explicit
    training arguments. Relative paths in JSONC are relative to that file.
    Results include run_dir, model_path, metrics, history (neural models),
    resolved config, predictions, and preprocessing state.
    """
    mod = _resolve_module(model)
    if isinstance(config, (str, Path)):
        config_path, config = config, None
    merged = resolve_config(mod, user_config=config, config_path=config_path)
    if "save_path" in train_kwargs:
        raise ValueError("run_training manages model saving; use output_dir instead of save_path")
    if data_dir is not None:
        merged.setdefault("data", {})["base_dir"] = str(Path(data_dir).resolve())
    elif "base_dir" in merged.get("data", {}):
        merged["data"]["base_dir"] = str(Path(merged["data"]["base_dir"]).resolve())
    merged["output_dir"] = str(Path(output_dir or merged.get("output_dir", "outputs")).resolve())
    run_dir = create_run_directory(merged["output_dir"], mod.name)
    suffix = ".pkl" if getattr(mod, "model_type", "keras") == "estimator" else ".keras"
    model_path = run_dir / ("model" + suffix)
    _write_json(run_dir / "config.json", merged)
    try:
        fitted, results = train(mod, config=merged, save_path=model_path, **train_kwargs)
        resolved = results["config"]
        _write_json(run_dir / "config.json", resolved)
        _write_json(run_dir / "run.json", {"model": mod.name, "model_file": model_path.name})
        state = results["state"]
        _write_json(run_dir / "selection.json", state.selection_report)
        _write_json(run_dir / "splits.json", state.split_info)
        names = resolved.get("data", {}).get("dof_names")
        metrics = regression_metrics(results["Y_test"], results["Y_pred"], names)
        if "test_loss" in results:
            metrics["raw_test_loss"] = results["test_loss"]
        metrics["raw"] = regression_metrics(results["Y_test"], results["Y_pred_raw"], names)
        _write_json(run_dir / "metrics.json", metrics)
        history = results.get("history")
        if history is not None:
            _write_json(run_dir / "history.json", history.history)
        _plots(run_dir, results, [item["name"] for item in metrics["per_dof"]])
    except Exception as exc:
        _write_json(run_dir / "failure.json", {"error": type(exc).__name__, "message": str(exc)})
        raise
    results.update(run_dir=run_dir, model_path=model_path, metrics=metrics)
    print(f"Run saved to: {run_dir}")
    print(f"Test RMSE: {metrics['aggregate']['rmse']:.6f}")
    return fitted, results

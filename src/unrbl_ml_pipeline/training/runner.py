"""Training runs with reproducible records and headless diagnostic plots."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .config import resolve_config
from .metrics import regression_metrics
from .progress import TrainingProgress
from .reporting import plot_predictions, write_performance_report
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

    plot_predictions(run_dir / "predictions.png", results["Y_test"], results["Y_pred"],
                     results["trial_id_test"], names)


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
    progress = TrainingProgress(merged.get("preprocessing_steps", []), save_model=True,
                                reports=True, verbose=train_kwargs.get("verbose", 1))
    _write_json(run_dir / "config.json", merged)
    try:
        fitted, results = train(mod, config=merged, save_path=model_path,
                                _progress=progress, **train_kwargs)
        progress.stage("Writing run records and diagnostic plots")
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
        performance_path = write_performance_report(
            run_dir, fitted, mod.name, getattr(mod, "model_type", "keras"),
            results, metrics, model_path,
        )
    except Exception as exc:
        _write_json(run_dir / "failure.json", {"error": type(exc).__name__, "message": str(exc)})
        raise
    results.update(run_dir=run_dir, model_path=model_path, metrics=metrics,
                   performance_path=performance_path)
    progress.message(f"Run saved to: {run_dir}")
    progress.message(f"Performance report: {performance_path}")
    progress.message(f"Test RMSE: {metrics['aggregate']['rmse']:.6f}")
    return fitted, results

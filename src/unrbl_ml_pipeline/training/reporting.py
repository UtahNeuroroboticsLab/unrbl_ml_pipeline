"""Human-readable performance readouts and grayscale diagnostic plots."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .metrics import regression_metrics


def _figure(*args, **kwargs):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(*args, **kwargs)
    FigureCanvasAgg(figure)
    return figure


def plot_predictions(path, truth, pred, trial_ids, names, title="Held-out predictions"):
    boundaries = np.r_[0, np.flatnonzero(trial_ids[1:] != trial_ids[:-1]) + 1, len(trial_ids)]
    stride = max(1, len(trial_ids) // 10000)
    x, indices = [], []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        selected = np.unique(np.r_[np.arange(start, end, stride), end - 1])
        x.extend(selected.tolist() + [np.nan])
        indices.extend(selected.tolist() + [-1])
    indices = np.asarray(indices)
    fig = _figure(figsize=(12, 2.5 * truth.shape[1]), layout="constrained")
    axes = fig.subplots(truth.shape[1], 1, squeeze=False)[:, 0]
    for d, ax in enumerate(axes):
        for values, label, style, color in ((truth, "Target", "-", "black"),
                                            (pred, "Prediction", "--", "0.4")):
            y = values[np.maximum(indices, 0), d].astype(float)
            y[indices < 0] = np.nan
            ax.plot(x, y, style, color=color, linewidth=0.8, label=label)
        ax.set_ylabel(names[d])
    axes[0].legend(frameon=False)
    axes[0].set_title(title + " (gaps mark trial/recording boundaries)")
    axes[-1].set_xlabel("Sample index in concatenated test segments")
    fig.savefig(path, dpi=150)
    fig.clear()


def _plot_scatter(path, truth, pred, names):
    ncols = min(3, len(names))
    nrows = (len(names) + ncols - 1) // ncols
    fig = _figure(figsize=(4 * ncols, 4 * nrows), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
    indices = np.linspace(0, len(truth) - 1, min(len(truth), 10000), dtype=int)
    for d, name in enumerate(names):
        ax = axes[d]
        ax.scatter(truth[indices, d], pred[indices, d], color="black", s=3, alpha=0.25)
        lo = float(min(truth[:, d].min(), pred[:, d].min()))
        hi = float(max(truth[:, d].max(), pred[:, d].max()))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        ax.plot([lo, hi], [lo, hi], "--", color="0.4", linewidth=1)
        ax.set(title=name, xlabel="Target", ylabel="Prediction", xlim=(lo, hi), ylim=(lo, hi))
    for ax in axes[len(names):]:
        ax.set_visible(False)
    fig.savefig(path, dpi=150)
    fig.clear()


def test_source_segments(results, model_type):
    """Map test prediction offsets to full source paths without basename collisions."""
    config = results["config"]
    windowed = model_type == "keras" and "window" in config["preprocessing_steps"]
    window = int(config.get("model", {}).get("window_size", 50)) if windowed else 1
    lag = int(config.get("model", {}).get("prediction_lag_samples", 0)) if windowed else 0
    segments, start = [], 0
    for file in results["state"].test_files:
        count = max(0, len(file.features) - window + 1 - lag)
        if count:
            segments.append({"start": start, "end": start + count,
                             "path": str(Path(file.path).resolve()), "trial_id": file.trial_id})
            start += count
    if start != len(results["Y_test"]):
        raise ValueError("Test source segments do not match the number of evaluated samples")
    return segments


def _text(value):
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def _number(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def _metrics_table(metrics):
    lines = ["| DOF | RMSE | MAE | R² | Corr |", "|---|---|---|---|---|"]
    for row in metrics["per_dof"]:
        lines.append(f"| {_text(row['name'])} | {_number(row['rmse'])} | {_number(row['mae'])} | "
                     f"{_number(row['r2'], 3)} | {_number(row['corr'], 3)} |")
    return lines


def _architecture(model):
    if not hasattr(model, "layers"):
        return type(model).__name__
    parts = []
    for layer in model.layers:
        details = []
        for key in ("filters", "kernel_size", "units"):
            if hasattr(layer, key):
                details.append(f"{key}={getattr(layer, key)}")
        activation = getattr(layer, "activation", None)
        if activation is not None:
            details.append(f"activation={getattr(activation, '__name__', type(activation).__name__)}")
        part = type(layer).__name__
        if details:
            part += "(" + ", ".join(details) + ")"
        parts.append(part)
    return " --> ".join(parts)


def write_performance_report(run_dir, model, model_name, model_type, results, metrics, model_path):
    """Write performance.md after creating the plots it embeds.

    The spotlight uses only already-evaluated held-out segments from the first
    source path in sorted order. No extra prediction or training is performed.
    """
    run_dir = Path(run_dir)
    config, state = results["config"], results["state"]
    names = [row["name"] for row in metrics["per_dof"]]
    segments = test_source_segments(results, model_type)
    sources = sorted({segment["path"] for segment in segments})
    spotlight_file = sources[0]
    spotlight_indices = np.concatenate([
        np.arange(segment["start"], segment["end"]) for segment in segments
        if segment["path"] == spotlight_file
    ])
    truth, pred = results["Y_test"], results["Y_pred"]
    spotlight = regression_metrics(truth[spotlight_indices], pred[spotlight_indices], names)
    _plot_scatter(run_dir / "predictions_scatter.png", truth, pred, names)
    plot_predictions(run_dir / "spotlight_predictions.png", truth[spotlight_indices],
                     pred[spotlight_indices], results["trial_id_test"][spotlight_indices], names,
                     title="Spotlight: " + Path(spotlight_file).name)
    spotlight_record = {"file": spotlight_file, "samples": len(spotlight_indices), "metrics": spotlight}
    (run_dir / "spotlight_metrics.json").write_text(
        json.dumps(spotlight_record, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )

    def link(label, path, image=False):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Report artifact is missing: {path}")
        target = quote(Path(os.path.relpath(path, run_dir)).as_posix(), safe="/.")
        return ("!" if image else "") + f"[{label}]({target})"

    def source_name(path):
        return _text(os.path.relpath(path, config["data"]["base_dir"])) if config.get("data", {}).get("base_dir") else _text(path)

    lines = [f"# Performance readout: {run_dir.name}", "", "## Data", "",
             f"- Selected recordings: {len(state.files)} of {len(state.selection_report)} matched",
             f"- Quality threshold (beta): {config.get('data', {}).get('beta_threshold', 0)}",
             f"- Split method: {', '.join(s for s in config['preprocessing_steps'] if s.startswith('split_'))}",
             f"- Random seed: {config.get('split', {}).get('random_state', 42)}",
             (f"- Recording date range: {config.get('data', {}).get('date_from') or 'unbounded'} to "
             f"{config.get('data', {}).get('date_to') or 'unbounded'}"), "",
             "| Split | Unique recordings | Segments | Windows/samples |", "|---|---|---|---|"]
    for split in ("train", "val", "test"):
        files = getattr(state, split + "_files")
        unique = len({str(Path(file.path).resolve()) for file in files})
        x = getattr(state, "X_" + split)
        count = len(x) if model_type == "keras" and x is not None else sum(len(file.features) for file in files)
        lines.append(f"| {split} | {unique} | {len(files)} | {count} |")
    lines += ["", "Segments are split trial/recording chunks. Recording counts can overlap across trial-level splits.", ""]
    if model_type == "keras":
        lines += [(f"Window size: {config.get('model', {}).get('window_size', 50)} samples. "
                  f"Prediction lag: {config.get('model', {}).get('prediction_lag_samples', 0)} samples."), ""]
    lines += ["## Output processing", ""]
    enabled = []
    for key in ("latching_filter", "deadband", "kalman_filter"):
        settings = config.get("postprocessing", {}).get(key, {})
        active = settings.get("enabled", False)
        if active:
            enabled.append(key)
        detail = ", ".join(f"{k}={v}" for k, v in settings.items() if k != "enabled")
        lines.append(f"- {key.replace('_', ' ')}: {'enabled' if active else 'disabled'}"
                     + (f" ({detail})" if active and detail else ""))
    if "use_deadband_activation_output" in config.get("architecture", {}):
        lines.append(f"- Model-output deadband: {'enabled' if config['architecture']['use_deadband_activation_output'] else 'disabled'}")
    lines += ["", ("Reported metrics and plots use postprocessed predictions." if enabled else
                   "Reported metrics and plots use raw model predictions."),
              "Raw metrics are also saved in " + link("metrics.json", run_dir / "metrics.json") +
              "; raw means before external postprocessing, including any transforms embedded in the model.", "",
              "## Test-set metrics (per DOF)", ""]
    lines += _metrics_table(metrics)
    aggregate = metrics["aggregate"]
    lines += ["", (f"**Mean per-DOF RMSE: {_number(aggregate['mean_dof_rmse'])}   "
              f"Mean correlation: {_number(aggregate['mean_correlation'], 3)}**"), "",
              (f"Overall RMSE across all samples and DOFs: {_number(aggregate['rmse'])}. "
              f"Overall MSE: {_number(aggregate['mse'])}. Overall MAE: {_number(aggregate['mae'])}."), "",
              ("N/A denotes an undefined statistic: R² for constant targets, or correlation when either signal is constant. "
              "Mean correlation excludes undefined DOFs. All metrics use every evaluated sample."), ""]
    if enabled:
        lines += ["### Raw metrics before external postprocessing", ""] + _metrics_table(metrics["raw"]) + [""]
    lines += ["## Model", "", f"- Registered model: {model_name}",
              "- Model file: " + link(Path(model_path).name, model_path),
              f"- Architecture: {_architecture(model)}"]
    if model_type == "keras":
        lines += [f"- Parameters: {model.count_params()}", f"- Loss: {config.get('loss', {}).get('type', 'mse')}",
                  f"- Optimizer: {config.get('optimizer', 'adam')}; initial learning rate: {config.get('learning_rate', 0.001)}",
                  f"- Batch size: {config.get('batch_size', 128)}"]
        layer_names = {layer.name for layer in model.layers}
        embedded = {'feature_normalization', 'denormalize_output'} <= layer_names
        lines.append("- Feature/target normalization embedded in model: " + ("yes" if embedded else "no"))
        history = results["history"].history
        lines.append(f"- Epochs completed: {len(history['loss'])} of {config.get('epochs', 200)} maximum")
        if history.get("val_loss"):
            best = int(np.argmin(history["val_loss"]))
            lines.append(f"- Best validation loss: {history['val_loss'][best]:.6f} at epoch {best + 1}")
    lines += ["", "## Predictions plot sources", "",
              (f"The time-series plot spans all {len(truth)} evaluated samples in concatenated test-segment order. "
              "Long traces are subsampled for rendering; gaps mark segment boundaries. "
              "Ranges below are zero-based, with the end index excluded."), "",
              "| # | Sample range [start, end) | KDF file | Segment |", "|---|---|---|---|"]
    for i, segment in enumerate(segments, 1):
        lines.append(f"| {i} | [{segment['start']}, {segment['end']}) | {source_name(segment['path'])} | {_text(segment['trial_id'])} |")
    lines += ["", "## Plots", ""]
    for label, filename in (("Training history", "loss.png"),
                            ("Predictions vs. ground truth (time series)", "predictions.png"),
                            ("Predictions vs. ground truth (scatter)", "predictions_scatter.png")):
        if (run_dir / filename).is_file():
            lines += [f"### {label}", "", link(label, run_dir / filename, image=True), ""]
    lines += ["Scatter plots show up to 10,000 evenly sampled points per DOF; the dashed line is perfect agreement.", "",
              "## Spotlight file", "", f"Source: {source_name(spotlight_file)}", "",
              (f"This spot-check uses only the {len(spotlight_indices)} held-out samples from this recording, "
              "with the same output processing as the overall report. The source is selected deterministically "
              "as the first test-recording path in sorted order; this is not a whole-file or independent evaluation."), ""]
    lines += _metrics_table(spotlight) + ["", link("Spotlight predictions", run_dir / "spotlight_predictions.png", image=True), "",
                                       "## Run records", ""]
    for filename in ("config.json", "run.json", "selection.json", "splits.json", "metrics.json", "history.json", "spotlight_metrics.json"):
        if (run_dir / filename).is_file():
            lines.append("- " + link(filename, run_dir / filename))
    path = run_dir / "performance.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

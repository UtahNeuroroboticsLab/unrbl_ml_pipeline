"""Bounded onset alignment of training cues after global lag correction."""

import os

import numpy as np

from .align_lag import slice_file
from .select_features import candidates

DEFAULTS = {
    "feature_indices": "emg",
    "max_residual_lag_samples": 10,
    "baseline_samples": 10,
    "threshold_multiplier": 3.0,
    "min_active_samples": 3,
}


def validate_steps(steps):
    """Fail before loading data if the optional step cannot run correctly."""
    if "align_trials" not in steps:
        return
    required = ("split_by_detected_trial_deadzone", "align_lag", "align_trials")
    if any(steps.count(s) != 1 for s in required):
        raise ValueError("align_trials requires exactly one split_by_detected_trial_deadzone, "
                         "align_lag, and align_trials, in that order")
    split, global_step, local_step = [steps.index(s) for s in required]
    if not split < global_step < local_step or any(
        s in ("select_features", "normalize", "window") for s in steps[:local_step]
    ) or any(s.startswith("split_") for s in steps[split + 1:]):
        raise ValueError("Place align_trials after the detected-trial split and align_lag, "
                         "before select_features, normalize, and window; do not split again")


def settings(config):
    cfg = dict(DEFAULTS, **config.get("trial_alignment", {}))
    for key in ("max_residual_lag_samples", "baseline_samples", "min_active_samples"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"trial_alignment.{key} must be a positive integer")
    value = cfg["threshold_multiplier"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError("trial_alignment.threshold_multiplier must be finite and positive")
    return cfg


def _mad(x, axis=None):
    return np.median(np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis)


def _detect(f, cfg, record, example):
    """Return a fallback reason, or None after recording an accepted onset."""
    n = len(f.features)
    info = f.trial_info
    start = np.flatnonzero(f.source_indices == info["cue_start"])
    end = np.flatnonzero(f.source_indices == info["cue_end"] - 1)
    if not len(start) or not len(end):
        return "incomplete_movement_context"
    cue, stop = int(start[0]), int(end[0]) + 1
    record["cue_onset"] = cue
    example["cue_onset"] = cue
    radius, baseline, persist = (cfg[k] for k in
        ("max_residual_lag_samples", "baseline_samples", "min_active_samples"))
    lo, hi = cue - radius, cue + radius
    if lo - baseline < 0:
        return "insufficient_baseline"
    if hi + persist > n:
        return "incomplete_search_context"
    idx = candidates(cfg["feature_indices"], f.features.shape[1])
    z = f.features[:, idx].astype(np.float64)
    usable = np.isfinite(z).all(axis=0)
    z = z[:, usable]
    if z.shape[1]:
        z = z[:, np.ptp(z, axis=0) > 0]
    record["usable_channels"] = int(z.shape[1])
    if not z.shape[1]:
        return "no_usable_channels"
    rest = z[lo - baseline:lo]
    center = np.median(rest, axis=0)
    scale = np.maximum(np.maximum(1.4826 * _mad(rest, axis=0),
                                  1e-6 * np.max(np.abs(z), axis=0)), 1e-12)
    trace = np.maximum((z - center) / scale, 0).mean(axis=1)
    base_trace = trace[lo - baseline:lo]
    threshold = float(np.median(base_trace) + cfg["threshold_multiplier"] *
                      max(1.4826 * float(_mad(base_trace)), 1e-6))
    if not np.isfinite(trace).all() or not np.isfinite(threshold):
        return "nonfinite_activation_trace"
    record["threshold"] = threshold
    example.update(trace=trace, threshold=threshold, search=(lo, hi),
                   baseline=(lo - baseline, lo))
    above = trace > threshold
    onsets = [i for i in range(lo, hi + 1)
              if not above[i - 1] and above[i:i + persist].all()]
    if not onsets:
        return "missing_onset"
    if len(onsets) > 1:
        return "ambiguous_onset"
    onset = onsets[0]
    shift = onset - cue
    record.update(emg_onset=onset, emg_source_index=int(f.feature_indices[onset]),
                  detected_residual_shift=shift)
    example["emg_onset"] = onset
    if abs(shift) == radius:
        return "search_bound_reached"
    if cue + shift < 0 or stop + shift > n:
        return "shift_truncates_movement"
    return None


def run(state, config):
    cfg = settings(config)
    if "alignment" not in state.fitted_preprocessing:
        raise ValueError("align_trials requires align_lag to run first")
    if (state.norm_stats is not None or "feature_selection" in state.fitted_preprocessing
            or state.X_train is not None):
        raise ValueError("align_trials must run before select_features, normalize, and window")
    if "trial_alignment" in state.fitted_preprocessing:
        raise ValueError("align_trials must run only once")
    for f in state.train_files:
        info = f.trial_info
        if info.get("kind") not in ("trial", "background") or (
            info.get("kind") == "trial" and not {"trial_id", "cue_start", "cue_end"} <= info.keys()
        ):
            raise ValueError("align_trials needs cue metadata from split_by_detected_trial_deadzone")
        # Validate indices even when a trial falls back before detection.
        candidates(cfg["feature_indices"], f.features.shape[1])
    global_lag = state.fitted_preprocessing["alignment"]["lag"]
    horizon = int(config.get("model", {}).get("prediction_lag_samples", 0))
    minimum = int(config.get("model", {}).get("window_size", 50)) + horizon
    records, files = [], []
    examples = {"accepted": [], "fallback": []}
    counts = {"corrected": 0, "unchanged": 0, "fallback": 0, "background": 0}
    for f in state.train_files:
        if f.trial_info["kind"] == "background":
            counts["background"] += 1
            files.append(f)
            continue
        record = {
            "path": os.path.abspath(f.path), "trial_id": f.trial_info["trial_id"],
            "segment_id": f.trial_id, "global_lag": global_lag,
            "cue_source_start": f.trial_info["cue_start"],
            "cue_source_end": f.trial_info["cue_end"],
            "cue_onset": None, "emg_onset": None, "emg_source_index": None,
            "detected_residual_shift": None, "residual_shift": 0,
            "status": "fallback", "reason": None, "cropped_samples": 0,
            "effective_target_offset": global_lag + horizon,
        }
        # Capture only a bounded number of full traces for diagnostic plotting.
        example = {}
        reason = _detect(f, cfg, record, example)
        shift = record["detected_residual_shift"] if reason is None else 0
        if reason is None and len(f.features) - abs(shift) < minimum:
            reason = "insufficient_window_samples"
        g = f
        if reason is None:
            a, b = max(0, shift), min(len(f.features), len(f.features) + shift)
            g = slice_file(f, a, b)
            # Features and timestamps stay on the globally aligned timeline.
            # Labels and their original source indices move together.
            g.kinematics = f.kinematics[a - shift:b - shift].copy()
            g.source_indices = f.source_indices[a - shift:b - shift].copy()
            record.update(residual_shift=shift, cropped_samples=abs(shift),
                          status="corrected" if shift else "unchanged",
                          effective_target_offset=global_lag - shift + horizon)
            g.trial_info.update(residual_shift=shift, global_lag=global_lag)
        record["reason"] = reason
        counts[record["status"]] += 1
        records.append(record)
        files.append(g)
        bucket = examples["fallback" if reason else "accepted"]
        if len(bucket) < 12:
            a = max(0, record["residual_shift"])
            example.update(record=record.copy(), before=f.kinematics.copy(),
                           after=g.kinematics.copy(), after_start=a)
            bucket.append(example)
    state.train_files = files
    # Interleave successes and fallbacks; fill spare slots from either category.
    state.alignment_examples = []
    for i in range(12):
        for bucket in examples.values():
            if i < len(bucket) and len(state.alignment_examples) < 12:
                state.alignment_examples.append(bucket[i])
    state.fitted_preprocessing["trial_alignment"] = {
        "settings": cfg, "train_only": True, "counts": counts, "trials": records,
        "onset_units": "segment samples after global alignment, before residual cropping",
        "offset_convention": "global_lag - residual_shift + prediction_lag_samples",
    }
    return state

"""Preprocessing step: build sliding causal windows from split files.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Converts per-file (features, kinematics) arrays into fixed-length
causal windows ``[N, window_size, num_features]``. Windows never cross
file/trial boundaries.
"""
from __future__ import annotations

import os

import numpy as np


def make_windows(files, window_size, prediction_lag_samples=0):
    """Build causal windows from a list of KdfFiles.

    Returns ``(X, Y, trial_id, source_file)``.
    """
    X_list, Y_list, trial_list, source_list = [], [], [], []
    for f in files:
        T = f.features.shape[0]
        n_windows = T - window_size + 1 - prediction_lag_samples
        if n_windows <= 0:
            continue
        X_f = np.lib.stride_tricks.sliding_window_view(
            f.features, window_shape=window_size, axis=0
        )[:n_windows]
        X_f = np.moveaxis(X_f, -1, 1)
        X_list.append(X_f)
        Y_list.append(
            f.kinematics[
                window_size - 1 + prediction_lag_samples:
                window_size - 1 + prediction_lag_samples + n_windows
            ]
        )
        source_name = os.path.basename(f.path)
        trial_list.extend([f.trial_id] * n_windows)
        source_list.extend([source_name] * n_windows)

    if not X_list:
        raise RuntimeError(
            "No windows produced -- window_size + prediction_lag_samples "
            "exceeds every file's sample count."
        )
    X = np.concatenate(X_list, axis=0).astype(np.float32, copy=False)
    Y = np.concatenate(Y_list, axis=0).astype(np.float32, copy=False)
    trial_id = np.array(trial_list)
    source_file = np.array(source_list)
    return X, Y, trial_id, source_file


def run(state, config):
    """Build causal windows for train/val/test splits.

    Config keys used:
        model.window_size           — window length (required)
        model.prediction_lag_samples — lag in samples (default 0)
    """
    model_cfg = config.get("model", {})
    window_size = int(model_cfg.get("window_size", 50))
    lag = int(model_cfg.get("prediction_lag_samples", 0))
    if window_size < 1 or lag < 0:
        raise ValueError("model.window_size must be positive and prediction_lag_samples nonnegative")

    for split_name in ("train", "val", "test"):
        files = getattr(state, f"{split_name}_files")
        if not files:
            continue
        X, Y, tid, sf = make_windows(files, window_size, prediction_lag_samples=lag)
        records, offset = [], 0
        for f in files:
            n = len(f.features) - window_size + 1 - lag
            if n <= 0:
                continue
            first = window_size - 1 + lag
            records.append({"start": offset, "end": offset+n, "path": os.path.abspath(f.path),
                "trial_id": f.trial_id, "segment_id": f"{os.path.abspath(f.path)}:{f.trial_id}:{int(f.source_indices[0])}",
                "target_indices": f.source_indices[first:first+n],
                "input_end_indices": f.feature_indices[window_size-1:window_size-1+n],
                "target_times": None if f.nip_time is None else f.nip_time[first:first+n],
                "nominal_interval": f.nominal_interval, "timing": f.timestamp_info.copy()})
            offset += n
        state.provenance[split_name] = records
        setattr(state, f"X_{split_name}", X)
        setattr(state, f"Y_{split_name}", Y)
        setattr(state, f"trial_id_{split_name}", tid)
        setattr(state, f"source_file_{split_name}", sf)

    return state

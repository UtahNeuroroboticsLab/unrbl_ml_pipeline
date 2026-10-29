"""Preprocessing step: split on kinematics-detected trial boundaries.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Detects movement episodes from the kinematics signal and splits
whole trials into train/val/test, stratified by movement type.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

from ._types import KdfFile
from .trial_detect import detect_trials_from_kinematics


def _chunk_of(f, start, end, suffix):
    return KdfFile(
        path=f.path,
        features=f.features[start:end].copy(),
        kinematics=f.kinematics[start:end].copy(),
        trial_id=f"{f.trial_id}_{suffix}",
    )


def _segment_chunks(f, tid):
    """Split a KdfFile into contiguous chunks wherever trial_id changes."""
    n = tid.shape[0]
    chunks = []
    start = 0
    for t in range(1, n + 1):
        if t == n or tid[t] != tid[start]:
            local_id = int(tid[start])
            chunk = _chunk_of(f, start, t, f"seg{start}")
            chunks.append((chunk, local_id > 0, local_id))
            start = t
    return chunks


def run(state, config):
    """Detect trials from kinematics and split them into train/val/test.

    Config keys used:
        split.test_size, split.val_size, split.random_state
        split.trial_detection.merge_gap, min_trial_len, min_peak,
            active_eps, non_trial_policy
    """
    split_cfg = config.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    val_size = float(split_cfg.get("val_size", 0.1))
    random_state = int(split_cfg.get("random_state", 42))

    cfg = split_cfg.get("trial_detection", {})
    non_trial_policy = str(cfg.get("non_trial_policy", "train")).lower()
    det_params = {
        "merge_gap": int(cfg.get("merge_gap", 0)),
        "min_trial_len": int(cfg.get("min_trial_len", 5)),
        "min_peak": float(cfg.get("min_peak", 0.5)),
        "active_eps": float(cfg.get("active_eps", 1e-9)),
    }

    trial_segments = {}
    trial_movement_types = {}
    background_segments = []
    next_gid = 1

    for f in state.files:
        detected = detect_trials_from_kinematics(f.kinematics, **det_params)
        tid = np.zeros(f.kinematics.shape[0], dtype=np.int64)
        local_to_global = {}
        for i, trial in enumerate(detected, 1):
            tid[trial["start"]:trial["end"]] = i
            local_to_global[i] = next_gid
            trial_movement_types[next_gid] = trial["movement_type"]
            next_gid += 1

        for chunk, is_trial, local_id in _segment_chunks(f, tid):
            if is_trial:
                gid = local_to_global[local_id]
                chunk.trial_id = f"trial_{gid}"
                trial_segments.setdefault(gid, []).append(chunk)
            else:
                background_segments.append(chunk)

    n_trials = len(trial_segments)
    if n_trials < 3:
        raise ValueError(
            f"split_by_detected_trial found only {n_trials} trial(s), need at least 3."
        )

    mt_groups = defaultdict(list)
    for gid, mt in trial_movement_types.items():
        mt_groups[mt].append(gid)

    train_ids, val_ids, test_ids = [], [], []
    for mt, gids in mt_groups.items():
        if len(gids) < 2:
            train_ids.extend(gids)
            continue
        tr, te = train_test_split(
            gids, test_size=test_size, random_state=random_state
        )
        if len(tr) >= 2:
            relative_val = val_size / (1.0 - test_size)
            tr, va = train_test_split(
                tr, test_size=relative_val, random_state=random_state
            )
            val_ids.extend(va)
        train_ids.extend(tr)
        test_ids.extend(te)

    state.train_files = [seg for gid in train_ids for seg in trial_segments[gid]]
    state.val_files = [seg for gid in val_ids for seg in trial_segments[gid]]
    state.test_files = [seg for gid in test_ids for seg in trial_segments[gid]]

    if non_trial_policy == "train":
        for i, seg in enumerate(background_segments):
            seg.trial_id = f"background_{i}"
        state.train_files = state.train_files + background_segments

    state.split_info = {
        "method": "by_detected_trial",
        "n_detected_trials": n_trials,
        "n_movement_types": len(mt_groups),
        "detection_params": det_params,
    }
    return state

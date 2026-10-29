"""Preprocessing step: split on real KEF-based trial boundaries.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Uses ground-truth trial metadata from sibling .kef files (not
kinematics-inferred boundaries). Files without a .kef are treated
as a single background segment.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.model_selection import train_test_split

from ._types import KdfFile
from .load_kdf import parse_kef, trial_id_from_kef


def _chunk_of(f, start, end, suffix):
    return KdfFile(
        path=f.path,
        features=f.features[start:end].copy(),
        kinematics=f.kinematics[start:end].copy(),
        trial_id=f"{f.trial_id}_{suffix}",
    )


def _segment_chunks(f, tid):
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
    """Split on real trial boundaries from .kef event files.

    Config keys used:
        split.test_size, split.val_size, split.random_state
        split.trial_detection.non_trial_policy  — "train" or "exclude"
    """
    split_cfg = config.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    val_size = float(split_cfg.get("val_size", 0.1))
    random_state = int(split_cfg.get("random_state", 42))

    cfg = split_cfg.get("trial_detection", {})
    non_trial_policy = str(cfg.get("non_trial_policy", "train")).lower()

    trial_segments = {}
    background_segments = []
    next_gid = 1

    for f in state.files:
        kef_path = os.path.splitext(f.path)[0] + ".kef"
        trials = parse_kef(kef_path) if os.path.exists(kef_path) else []
        if trials and f.nip_time is not None:
            tid = trial_id_from_kef(f.nip_time, trials)
        else:
            tid = np.zeros(f.features.shape[0], dtype=np.int64)

        local_to_global = {}
        for chunk, is_trial, local_id in _segment_chunks(f, tid):
            if is_trial:
                if local_id not in local_to_global:
                    local_to_global[local_id] = next_gid
                    next_gid += 1
                gid = local_to_global[local_id]
                chunk.trial_id = f"trial_{gid}"
                trial_segments.setdefault(gid, []).append(chunk)
            else:
                background_segments.append(chunk)

    n_real_trials = len(trial_segments)
    if n_real_trials == 0:
        raise ValueError(
            f"split_by_trial found NO real trial metadata (.kef files) "
            f"for any of the {len(state.files)} kept file(s). Use a different split method."
        )
    if n_real_trials < 3:
        raise ValueError(
            f"split_by_trial found only {n_real_trials} trial(s), need at least 3."
        )

    global_ids = sorted(trial_segments.keys())
    train_ids, test_ids = train_test_split(
        global_ids, test_size=test_size, random_state=random_state
    )
    relative_val = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        train_ids, test_size=relative_val, random_state=random_state
    )

    state.train_files = [seg for gid in train_ids for seg in trial_segments[gid]]
    state.val_files = [seg for gid in val_ids for seg in trial_segments[gid]]
    state.test_files = [seg for gid in test_ids for seg in trial_segments[gid]]

    if non_trial_policy == "train":
        for i, seg in enumerate(background_segments):
            seg.trial_id = f"background_{i}"
        state.train_files = state.train_files + background_segments

    state.split_info = {
        "method": "by_trial",
        "n_real_trials": n_real_trials,
        "non_trial_policy": non_trial_policy,
    }
    return state

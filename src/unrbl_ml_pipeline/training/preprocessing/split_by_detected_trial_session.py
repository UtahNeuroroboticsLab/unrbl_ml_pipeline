"""Preprocessing step: detected-trial + session-level split.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Detects trials with deadzone, groups by recording session, and
assigns whole sessions to train/val/test so no recording day appears
in more than one split.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

from ._types import KdfFile
from .trial_detect import detect_trials_from_kinematics, extend_trials_with_deadzone

_SESSION_DATE_RE = re.compile(r"(\d{8})[-_]\d{6}")


def _session_key(path):
    name = os.path.basename(path)
    m = _SESSION_DATE_RE.search(name)
    return m.group(1) if m else name


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
    """Detect trials with deadzone, group by session, split sessions.

    Config keys: same as split_by_detected_trial.
    """
    split_cfg = config.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    val_size = float(split_cfg.get("val_size", 0.1))
    random_state = int(split_cfg.get("random_state", 42))

    cfg = split_cfg.get("trial_detection", {})
    det_params = {
        "merge_gap": int(cfg.get("merge_gap", 0)),
        "min_trial_len": int(cfg.get("min_trial_len", 5)),
        "min_peak": float(cfg.get("min_peak", 0.5)),
        "active_eps": float(cfg.get("active_eps", 1e-9)),
    }

    session_trial_segs = defaultdict(list)
    session_bg_segs = defaultdict(list)

    for f in state.files:
        sk = _session_key(f.path)
        detected = detect_trials_from_kinematics(f.kinematics, **det_params)
        detected = extend_trials_with_deadzone(detected, f.kinematics.shape[0])
        tid = np.zeros(f.kinematics.shape[0], dtype=np.int64)
        for i, trial in enumerate(detected, 1):
            tid[trial["start"]:trial["end"]] = i

        for chunk, is_trial, local_id in _segment_chunks(f, tid):
            if is_trial:
                chunk.trial_id = f"trial_{sk}_{local_id}"
                session_trial_segs[sk].append(chunk)
            else:
                chunk.trial_id = f"background_{sk}"
                session_bg_segs[sk].append(chunk)

    all_sessions = sorted(set(session_trial_segs) | set(session_bg_segs))
    if len(all_sessions) < 3:
        raise ValueError(
            f"split_by_detected_trial_session needs at least 3 sessions, "
            f"got {len(all_sessions)} ({all_sessions})."
        )

    train_keys, test_keys = train_test_split(
        all_sessions, test_size=test_size, random_state=random_state
    )
    relative_val = val_size / (1.0 - test_size)
    train_keys, val_keys = train_test_split(
        train_keys, test_size=relative_val, random_state=random_state
    )

    def _gather(keys):
        out = []
        for k in keys:
            out.extend(session_trial_segs.get(k, []))
            out.extend(session_bg_segs.get(k, []))
        return out

    state.train_files = _gather(train_keys)
    state.val_files = _gather(val_keys)
    state.test_files = _gather(test_keys)
    state.split_info = {
        "method": "by_detected_trial_session",
        "train_sessions": train_keys,
        "val_sessions": val_keys,
        "test_sessions": test_keys,
        "detection_params": det_params,
    }
    return state

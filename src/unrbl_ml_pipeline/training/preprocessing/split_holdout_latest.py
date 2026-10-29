"""Preprocessing step: hold out the newest recording as test.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

The newest .kdf file (by YYYYMMDD-HHMMSS timestamp in its filename)
becomes the unsplit test set. Training uses only files recorded before
it. Supports "single" (whole-file fit, no val) and "aggregate"
(detected-trial split of all prior files into train/val).
"""
from __future__ import annotations

import os
import re

from .quality_filter import correlation_quality_score
from .split_by_detected_trial_deadzone import run as _detected_trial_run

_SESSION_TIMESTAMP_RE = re.compile(r"(\d{8})[-_](\d{6})")


def _session_timestamp(path):
    m = _SESSION_TIMESTAMP_RE.search(os.path.basename(path))
    return f"{m.group(1)}-{m.group(2)}" if m else None


def run(state, config):
    """Hold out the newest recording as the test set.

    Config keys used:
        split.holdout_latest.test_file        — basename override
        split.holdout_latest.mode             — "single" or "aggregate"
        split.holdout_latest.exclude_test_day — bool
        split.val_size, split.random_state
        data.beta_threshold
    """
    split_cfg = config.get("split", {})
    hl = split_cfg.get("holdout_latest", {})
    mode = str(hl.get("mode", "aggregate")).lower()
    beta = float(config.get("data", {}).get("beta_threshold", 0.0))

    stamped = []
    for f in state.files:
        ts = _session_timestamp(f.path)
        if ts is not None:
            score, _ = correlation_quality_score(f.features, f.kinematics)
            stamped.append((f, score, ts))

    if not stamped:
        raise ValueError(
            "holdout_latest found no files with a YYYYMMDD-HHMMSS timestamp."
        )

    test_name = hl.get("test_file")
    if test_name:
        matches = [t for t in stamped if os.path.basename(t[0].path) == test_name]
        if not matches:
            raise ValueError(
                f"holdout_latest.test_file '{test_name}' not found."
            )
        test_f, _test_s, test_ts = matches[0]
    else:
        test_f, _test_s, test_ts = max(stamped, key=lambda t: t[2])

    test_day = test_ts[:8]
    exclude_test_day = bool(hl.get("exclude_test_day", False))

    cands = [
        (f, s, ts) for f, s, ts in stamped
        if ts < test_ts and s >= beta
        and not (exclude_test_day and ts[:8] == test_day)
    ]
    cands.sort(key=lambda t: t[2])
    if not cands:
        raise ValueError(
            f"holdout_latest has no training files recorded before '{os.path.basename(test_f.path)}'."
        )

    cand_files = [f for f, _, _ in cands]
    if mode == "single":
        state.train_files = cand_files
        state.val_files = []
    else:
        from ._types import PipelineState as PS
        sub = PS(files=cand_files)
        sub_cfg = dict(config)
        sub_cfg.setdefault("split", {})
        sub_cfg["split"]["test_size"] = 0.0
        sub = _detected_trial_run(sub, sub_cfg)
        state.train_files = sub.train_files
        state.val_files = sub.val_files

    state.test_files = [test_f]
    state.split_info = {
        "method": "holdout_latest",
        "mode": mode,
        "test_file": os.path.basename(test_f.path),
        "test_timestamp": test_ts,
        "exclude_test_day": exclude_test_day,
    }
    return state

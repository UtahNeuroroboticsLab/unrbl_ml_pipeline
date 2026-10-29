"""Helper: detect trial boundaries from kinematics signals.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Used by the ``split_by_detected_trial*`` step modules.
"""
from __future__ import annotations

import numpy as np


def detect_trials_from_kinematics(
    kinematics,
    merge_gap=0,
    min_trial_len=5,
    min_peak=0.5,
    active_eps=1e-9,
):
    """Detect movement episodes by thresholding non-zero activity.

    Returns list of dicts with ``start``, ``end`` (exclusive), ``movement_type``.
    """
    if kinematics.size == 0:
        return []
    active = np.max(np.abs(kinematics), axis=1) > active_eps
    changes = np.diff(active.astype(np.int8), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if len(starts) == 0:
        return []

    merged_s, merged_e = [int(starts[0])], [int(ends[0])]
    for s, e in zip(starts[1:], ends[1:]):
        if int(s) - merged_e[-1] <= merge_gap:
            merged_e[-1] = int(e)
        else:
            merged_s.append(int(s))
            merged_e.append(int(e))

    trials = []
    for s, e in zip(merged_s, merged_e):
        if e - s < min_trial_len:
            continue
        seg = kinematics[s:e]
        if np.max(np.abs(seg)) < min_peak:
            continue
        mt = _infer_movement_type(seg, min_peak=min_peak)
        if not mt:
            continue
        trials.append({"start": s, "end": e, "movement_type": mt})
    return trials


def extend_trials_with_deadzone(trials, total_length):
    """Extend each trial's end to the next trial's start (capturing the rest period)."""
    if not trials:
        return trials
    extended = []
    for i, trial in enumerate(trials):
        new_end = trials[i + 1]["start"] if i + 1 < len(trials) else total_length
        extended.append({**trial, "end": new_end})
    return extended


def _infer_movement_type(kinematics_segment, min_peak=0.5):
    """Return a tuple of (dof, sign) pairs identifying which DOFs are active."""
    movement = []
    for dof in range(kinematics_segment.shape[1]):
        col = kinematics_segment[:, dof]
        if np.max(col) >= min_peak:
            movement.append((dof, 1))
        if np.min(col) <= -min_peak:
            movement.append((dof, -1))
    return tuple(movement)

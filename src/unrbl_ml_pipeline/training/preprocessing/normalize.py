"""Preprocessing step: compute normalization statistics from the train split.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Computes mean/std from ``train_files`` only so that val/test are
never leaked into normalization. Stats are stored on ``state.norm_stats``
and typically baked into the model's normalization layer rather than
applied to the raw arrays.
"""
from __future__ import annotations

import numpy as np

from ._types import NormStats


def compute_norm_stats(train_files):
    """Compute NormStats from a list of KdfFiles."""
    all_feat = np.concatenate([f.features for f in train_files], axis=0)
    all_kin = np.concatenate([f.kinematics for f in train_files], axis=0)
    feat_std = all_feat.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    kin_std = all_kin.std(axis=0)
    kin_std[kin_std < 1e-8] = 1.0
    return NormStats(
        feat_mean=all_feat.mean(axis=0),
        feat_std=feat_std,
        kin_mean=all_kin.mean(axis=0),
        kin_std=kin_std,
    )


def run(state, config):
    """Compute normalization statistics from the train split.

    No config keys used; operates on ``state.train_files``.
    """
    if not state.train_files:
        raise ValueError("normalize: no train_files to compute stats from.")
    state.norm_stats = compute_norm_stats(state.train_files)
    return state

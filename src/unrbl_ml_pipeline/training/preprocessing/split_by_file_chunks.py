"""Preprocessing step: chronological within-file chunk split.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Every file contributes a contiguous chronological chunk to each of
train (earliest), val (middle), and test (latest).
"""
from __future__ import annotations

from ._types import KdfFile


def _chunk_of(f, start, end, suffix):
    return KdfFile(
        path=f.path,
        features=f.features[start:end].copy(),
        kinematics=f.kinematics[start:end].copy(),
        trial_id=f"{f.trial_id}_{suffix}",
    )


def run(state, config):
    """Split every file into chronological train/val/test chunks.

    Config keys used:
        split.test_size, split.val_size
    """
    split_cfg = config.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    val_size = float(split_cfg.get("val_size", 0.1))

    train_files, val_files, test_files = [], [], []
    for f in state.files:
        T = f.features.shape[0]
        n_test = round(T * test_size)
        n_val = round(T * val_size)
        n_train = T - n_val - n_test
        if n_train <= 0:
            raise ValueError(
                f"{f.path}: only {T} samples, too short for chunk split."
            )
        train_files.append(_chunk_of(f, 0, n_train, "train"))
        if n_val > 0:
            val_files.append(_chunk_of(f, n_train, n_train + n_val, "val"))
        if n_test > 0:
            test_files.append(_chunk_of(f, n_train + n_val, T, "test"))

    state.train_files = train_files
    state.val_files = val_files
    state.test_files = test_files
    return state

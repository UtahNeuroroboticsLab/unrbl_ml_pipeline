"""Preprocessing step: random whole-file train/val/test split.

Author: Leonardo M. Ferrisi
Date: 2026-10-29
"""
from __future__ import annotations

from sklearn.model_selection import train_test_split


def run(state, config):
    """Randomly assign whole files to train/val/test.

    Config keys used:
        split.test_size     — fraction for test (default 0.2)
        split.val_size      — fraction for val  (default 0.1)
        split.random_state  — RNG seed (default 42)
    """
    split_cfg = config.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    val_size = float(split_cfg.get("val_size", 0.1))
    random_state = int(split_cfg.get("random_state", 42))

    files = state.files
    if len(files) < 3:
        raise ValueError(
            f"Need at least 3 kdf files for train/val/test splits, got {len(files)}. "
            "Loosen beta_threshold or use split_by_file_chunks."
        )
    train, test = train_test_split(
        files, test_size=test_size, random_state=random_state
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train, test_size=relative_val, random_state=random_state
    )
    state.train_files = train
    state.val_files = val
    state.test_files = test
    return state

"""Preprocessing step: split by recording session (YYYYMMDD date).

Author: Leonardo M. Ferrisi
Date: 2026-10-29
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

from sklearn.model_selection import train_test_split

_SESSION_DATE_RE = re.compile(r"(\d{8})[-_]\d{6}")


def session_key(path):
    """Extract YYYYMMDD session key from a .kdf filename."""
    name = os.path.basename(path)
    m = _SESSION_DATE_RE.search(name)
    return m.group(1) if m else name


def run(state, config):
    """Group files by session date and assign whole sessions to splits.

    Config keys used:
        split.test_size, split.val_size, split.random_state
    """
    split_cfg = config.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    val_size = float(split_cfg.get("val_size", 0.1))
    random_state = int(split_cfg.get("random_state", 42))

    groups = defaultdict(list)
    for f in state.files:
        groups[session_key(f.path)].append(f)

    session_keys = sorted(groups.keys())
    if len(session_keys) < 3:
        raise ValueError(
            f"Need at least 3 distinct sessions for by_session split, "
            f"got {len(session_keys)} ({session_keys})."
        )

    train_keys, test_keys = train_test_split(
        session_keys, test_size=test_size, random_state=random_state
    )
    relative_val = val_size / (1.0 - test_size)
    train_keys, val_keys = train_test_split(
        train_keys, test_size=relative_val, random_state=random_state
    )

    state.train_files = [f for k in train_keys for f in groups[k]]
    state.val_files = [f for k in val_keys for f in groups[k]]
    state.test_files = [f for k in test_keys for f in groups[k]]
    state.split_info = {
        "method": "by_session",
        "train_sessions": train_keys,
        "val_sessions": val_keys,
        "test_sessions": test_keys,
    }
    return state

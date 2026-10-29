"""Shared data types for preprocessing steps.

Author: Leonardo M. Ferrisi
Date: 2026-10-29
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class KdfFile:
    """One loaded .kdf recording."""
    path: str
    features: np.ndarray       # [T, num_features]
    kinematics: np.ndarray     # [T, num_dofs]
    nip_time: np.ndarray | None = None  # [T]
    trial_id: str = field(default="")

    def __post_init__(self):
        if not self.trial_id:
            self.trial_id = os.path.splitext(os.path.basename(self.path))[0]


@dataclass
class NormStats:
    """Feature and kinematics normalization statistics (from train split)."""
    feat_mean: np.ndarray
    feat_std: np.ndarray
    kin_mean: np.ndarray
    kin_std: np.ndarray


@dataclass
class PipelineState:
    """Mutable bag passed between preprocessing steps.

    Each step reads the fields it needs and writes the fields it produces.
    """
    # Raw loaded files (before splitting)
    files: list[Any] = field(default_factory=list)
    rejected: list[Any] = field(default_factory=list)

    # After splitting
    train_files: list[Any] = field(default_factory=list)
    val_files: list[Any] = field(default_factory=list)
    test_files: list[Any] = field(default_factory=list)
    split_info: dict[str, Any] = field(default_factory=dict)

    # After normalization
    norm_stats: NormStats | None = None

    # After windowing
    X_train: np.ndarray | None = None
    Y_train: np.ndarray | None = None
    X_val: np.ndarray | None = None
    Y_val: np.ndarray | None = None
    X_test: np.ndarray | None = None
    Y_test: np.ndarray | None = None
    trial_id_train: np.ndarray | None = None
    trial_id_val: np.ndarray | None = None
    trial_id_test: np.ndarray | None = None
    source_file_train: np.ndarray | None = None
    source_file_val: np.ndarray | None = None
    source_file_test: np.ndarray | None = None

    # After trajectory/stateful chunking (Keras Sequences)
    train_sequence: Any = None
    val_sequence: Any = None

"""Preprocessing step: build trajectory batch sequences for trajectory loss.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Partitions windowed data into non-overlapping trajectory chunks and
wraps them in a Keras Sequence for trajectory-loss training
(``predicted_trajectory_loss_fast``).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def nonoverlapping_trajectory_starts(trial_id, trajectory_length):
    """Return row indices where a trajectory chunk begins.

    Each contiguous run of equal ``trial_id`` values is partitioned into
    non-overlapping chunks of ``trajectory_length``. Remainders shorter
    than ``trajectory_length`` are dropped.
    """
    trial_id = np.asarray(trial_id).reshape(-1)
    trajectory_length = int(trajectory_length)
    n = trial_id.shape[0]
    starts = []
    i = 0
    while i < n:
        j = i
        while j < n and trial_id[j] == trial_id[i]:
            j += 1
        run_len = j - i
        n_chunks = run_len // trajectory_length
        for k in range(n_chunks):
            starts.append(i + k * trajectory_length)
        i = j
    return np.array(starts, dtype=np.int64)


class TrajectoryBatchSequence(tf.keras.utils.Sequence):
    """Keras Sequence that yields trajectory-chunk batches on demand."""

    def __init__(self, X, Y, starts, trajectory_length, batch_size, shuffle, seed=0):
        super().__init__()
        self.X = X
        self.Y = Y
        self.starts = np.asarray(starts, dtype=np.int64).reshape(-1)
        self.trajectory_length = int(trajectory_length)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(int(seed))
        if self.starts.size == 0:
            raise ValueError(
                "No trajectory chunks produced -- temporal_batch_length is "
                "probably longer than every contiguous trial run."
            )
        self.offsets = np.arange(self.trajectory_length, dtype=np.int64)
        self.order = np.arange(self.starts.size, dtype=np.int64)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.starts.size / self.batch_size))

    def __getitem__(self, batch_index):
        lo = int(batch_index) * self.batch_size
        hi = min(lo + self.batch_size, self.starts.size)
        selected = self.starts[self.order[lo:hi]]
        rows = selected[:, None] + self.offsets[None, :]
        return self.X[rows], self.Y[rows]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def run(state, config):
    """Build TrajectoryBatchSequence for train (and optionally val).

    Config keys used:
        loss.temporal_batch_length  — trajectory chunk length (required)
        training.batch_size         — batch size (default 32)
        training.seed               — RNG seed (default 0)
    """
    loss_cfg = config.get("loss", {})
    traj_len = int(loss_cfg.get("temporal_batch_length", 10))
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    seed = int(config.get("training", {}).get("seed", 0))

    if state.X_train is not None and state.trial_id_train is not None:
        starts = nonoverlapping_trajectory_starts(state.trial_id_train, traj_len)
        state.train_sequence = TrajectoryBatchSequence(
            state.X_train, state.Y_train, starts, traj_len,
            batch_size, shuffle=True, seed=seed,
        )

    if state.X_val is not None and state.trial_id_val is not None:
        starts = nonoverlapping_trajectory_starts(state.trial_id_val, traj_len)
        if starts.size > 0:
            state.val_sequence = TrajectoryBatchSequence(
                state.X_val, state.Y_val, starts, traj_len,
                batch_size, shuffle=False, seed=seed,
            )

    return state

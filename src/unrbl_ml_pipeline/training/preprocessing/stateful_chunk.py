"""Preprocessing step: build StatefulChunkSequence for CLSTM TBPTT training.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Splits each trial's windows into non-overlapping chunks for truncated
backpropagation through time. The LSTM hidden state (h, c) is carried
forward within a trial and reset at trial boundaries.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


class StatefulChunkSequence(tf.keras.utils.Sequence):
    """Keras Sequence for stateful LSTM training via TBPTT.

    Yields::

        X = {"trajectory_features": [B, chunk_len, win, feat],
             "h0": [B, units], "c0": [B, units]}
        Y = [B, chunk_len, dofs]
    """

    def __init__(
        self, X, Y, trial_id, chunk_length, lstm_units,
        model=None, batch_size=8, shuffle_trials=True, seed=0,
    ):
        super().__init__()
        self.X = X
        self.Y = Y
        self.trial_id = np.asarray(trial_id).reshape(-1)
        self.chunk_length = int(chunk_length)
        self.lstm_units = int(lstm_units)
        self.model = model
        self.batch_size = int(batch_size)
        self.shuffle_trials = bool(shuffle_trials)
        self.rng = np.random.default_rng(int(seed))

        self.trials = []
        i = 0
        n = len(self.trial_id)
        while i < n:
            j = i
            while j < n and self.trial_id[j] == self.trial_id[i]:
                j += 1
            self.trials.append((i, j - i))
            i = j

        self.chunks = []
        for trial_idx, (start, length) in enumerate(self.trials):
            n_full = length // self.chunk_length
            for c in range(n_full):
                self.chunks.append(
                    (trial_idx, c * self.chunk_length, self.chunk_length)
                )
            remainder = length % self.chunk_length
            if remainder >= 2:
                self.chunks.append(
                    (trial_idx, n_full * self.chunk_length, remainder)
                )

        if not self.chunks:
            raise ValueError(
                "No stateful chunks produced -- chunk_length is probably "
                "longer than every contiguous trial."
            )

        self.chunk_h0 = np.zeros(
            (len(self.chunks), self.lstm_units), dtype=np.float32
        )
        self.chunk_c0 = np.zeros(
            (len(self.chunks), self.lstm_units), dtype=np.float32
        )
        self._build_batch_order()

    def set_model(self, model):
        """Set the single-window stateful model after construction."""
        self.model = model

    def _build_batch_order(self):
        trial_chunks = {}
        for ci, (trial_idx, _, _) in enumerate(self.chunks):
            trial_chunks.setdefault(trial_idx, []).append(ci)
        trial_order = list(trial_chunks.keys())
        if self.shuffle_trials:
            self.rng.shuffle(trial_order)
        self.ordered_chunk_indices = []
        for tidx in trial_order:
            self.ordered_chunk_indices.extend(trial_chunks[tidx])
        self.ordered_chunk_indices = np.array(
            self.ordered_chunk_indices, dtype=np.int64
        )

    def _refresh_initial_states(self):
        """Forward-pass the stateful model to compute per-chunk (h, c)."""
        if self.model is None:
            self.chunk_h0[:] = 0.0
            self.chunk_c0[:] = 0.0
            return

        trial_chunk_map = {}
        for ci, (trial_idx, offset, _) in enumerate(self.chunks):
            trial_chunk_map.setdefault(trial_idx, []).append((ci, offset))

        for trial_idx, (trial_start, trial_length) in enumerate(self.trials):
            if trial_idx not in trial_chunk_map:
                continue
            chunk_infos = sorted(trial_chunk_map[trial_idx], key=lambda x: x[1])
            h = tf.zeros((1, self.lstm_units), dtype=tf.float32)
            c = tf.zeros((1, self.lstm_units), dtype=tf.float32)
            ci_iter = iter(chunk_infos)
            next_ci, next_offset = next(ci_iter, (None, None))

            for t in range(trial_length):
                if next_ci is not None and t == next_offset:
                    self.chunk_h0[next_ci] = h.numpy()[0]
                    self.chunk_c0[next_ci] = c.numpy()[0]
                    next_ci, next_offset = next(ci_iter, (None, None))

                x = tf.constant(self.X[trial_start + t:trial_start + t + 1])
                _, h, c = self.model(
                    {"features": x, "lstm_h_in": h, "lstm_c_in": c},
                    training=False,
                )

    def __len__(self):
        return int(np.ceil(len(self.ordered_chunk_indices) / self.batch_size))

    def __getitem__(self, batch_index):
        lo = int(batch_index) * self.batch_size
        hi = min(lo + self.batch_size, len(self.ordered_chunk_indices))
        selected = self.ordered_chunk_indices[lo:hi]

        B = hi - lo
        window_size = self.X.shape[1]
        num_features = self.X.shape[2]
        num_dofs = self.Y.shape[1]

        X_batch = np.zeros(
            (B, self.chunk_length, window_size, num_features), dtype=np.float32
        )
        Y_batch = np.zeros(
            (B, self.chunk_length, num_dofs), dtype=np.float32
        )
        H0 = np.zeros((B, self.lstm_units), dtype=np.float32)
        C0 = np.zeros((B, self.lstm_units), dtype=np.float32)

        for b, ci in enumerate(selected):
            trial_idx, offset, chunk_len = self.chunks[ci]
            trial_start = self.trials[trial_idx][0]
            start = trial_start + offset
            X_batch[b, :chunk_len] = self.X[start:start + chunk_len]
            Y_batch[b, :chunk_len] = self.Y[start:start + chunk_len]
            H0[b] = self.chunk_h0[ci]
            C0[b] = self.chunk_c0[ci]

        return (
            {"trajectory_features": X_batch, "h0": H0, "c0": C0},
            Y_batch,
        )

    def on_epoch_end(self):
        self._refresh_initial_states()
        self._build_batch_order()


def run(state, config):
    """Build StatefulChunkSequence for CLSTM TBPTT training.

    Config keys used:
        model.stateful_chunk_length  — TBPTT chunk length (required)
        model.lstm_units             — LSTM hidden size (required)
        training.batch_size          — batch size (default 8)
        training.seed                — RNG seed (default 0)
    """
    model_cfg = config.get("model", {})
    chunk_length = int(model_cfg.get("stateful_chunk_length", 64))
    lstm_units = int(model_cfg.get("lstm_units", 128))
    batch_size = int(config.get("training", {}).get("batch_size", 8))
    seed = int(config.get("training", {}).get("seed", 0))

    if state.X_train is not None and state.trial_id_train is not None:
        state.train_sequence = StatefulChunkSequence(
            state.X_train, state.Y_train, state.trial_id_train,
            chunk_length, lstm_units,
            batch_size=batch_size, shuffle_trials=True, seed=seed,
        )

    if state.X_val is not None and state.trial_id_val is not None:
        state.val_sequence = StatefulChunkSequence(
            state.X_val, state.Y_val, state.trial_id_val,
            chunk_length, lstm_units,
            batch_size=batch_size, shuffle_trials=False, seed=seed,
        )

    return state

"""Chronological streams, explicit provenance and finite-size host batches."""

import numpy as np


def validate_metadata(state, config):
    for split in ("train", "val", "test"):
        x = getattr(state, "X_" + split)
        records = state.provenance.get(split, [])
        if x is None or not len(x) or not records:
            raise ValueError(
                f"Mamba requires nonempty {split} windows and source provenance"
            )
        count = 0
        for record in records:
            a, b = record["start"], record["end"]
            times, interval = record["target_times"], record["nominal_interval"]
            if (
                a != count
                or times is None
                or interval is None
                or interval <= 0
                or not np.isfinite(times).all()
            ):
                raise ValueError(f"{split}: missing or invalid chronological metadata")
            if np.any(np.diff(record["target_indices"]) != 1) or np.any(
                np.diff(record["input_end_indices"]) != 1
            ):
                raise ValueError(
                    f"{split}: nonconsecutive source indices require separate segments"
                )
            dt = np.diff(np.asarray(times, dtype=np.float64))
            if np.any(dt <= 0) or np.any(dt > 1.5 * interval):
                raise ValueError(
                    f"{split}: timestamp gaps require align_lag before windowing"
                )
            if b - a > 1 and not np.array_equal(x[a : b - 1, 1:], x[a + 1 : b, :-1]):
                raise ValueError(
                    f"{split}: chronological windows do not overlap consistently"
                )
            count = b
        if count != len(x):
            raise ValueError(f"{split}: provenance does not cover windows exactly once")


def batches(state, split, streams=4, chunk=16, rng=None):
    x, y = getattr(state, "X_" + split), getattr(state, "Y_" + split)
    records = state.provenance[split]
    order = np.arange(len(records)) if rng is None else rng.permutation(len(records))
    for start in range(0, len(order), streams):
        group = [records[i] for i in order[start : start + streams]]
        longest = max(r["end"] - r["start"] for r in group)
        for pos in range(0, longest, chunk):
            length = min(chunk, longest - pos)
            xx = np.zeros((len(group), length) + x.shape[1:], np.float32)
            yy = np.zeros((len(group), length, y.shape[-1]), np.float32)
            mask = np.zeros((len(group), length), bool)
            edges = np.zeros((len(group), max(0, length - 1)), bool)
            ids = np.full((len(group), length), -1, np.int64)
            for row, r in enumerate(group):
                a, end = r["start"] + pos, r["end"]
                n = max(0, min(length, end - a))
                if n:
                    xx[row, :n], yy[row, :n] = x[a : a + n], y[a : a + n]
                    mask[row, :n] = True
                    ids[row, :n] = np.arange(a, a + n)
                    times = np.asarray(r["target_times"], dtype=np.float64)[
                        pos : pos + n
                    ]
                    edges[row, : max(0, n - 1)] = (
                        np.abs(np.diff(times) - r["nominal_interval"])
                        <= 0.1 * r["nominal_interval"]
                    )
            yield xx, yy, mask, edges, ids, pos == 0


def predict(model, state, split, config):
    out = np.empty_like(getattr(state, "Y_" + split))
    seen = np.zeros(len(out), int)
    if model.state_mode == "reset":
        x = getattr(state, "X_" + split)
        size = int(config.get("batch_size", 64))
        for a in range(0, len(x), size):
            out[a : a + size] = model.predict_windows(x[a : a + size]).numpy()
            seen[a : a + size] += 1
    else:
        runtime = None
        for x, _, mask, _, ids, reset in batches(
            state, split, config.get("streams", 4), config.get("tbptt_steps", 16)
        ):
            if reset:
                runtime = model.prefill(x[:, 0, :-1])
            p, runtime = model.step(x[:, :, -1], runtime)
            out[ids[mask]] = p.numpy()[mask]
            seen[ids[mask]] += 1
    if not (seen == 1).all() or not np.isfinite(out).all():
        raise ValueError(
            "Evaluation must produce exactly one finite prediction per sample"
        )
    return out

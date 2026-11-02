"""Training-only lag scoring with non-wrapping, boundary-isolated alignment."""

import numpy as np

from ._types import KdfFile


def slice_file(f, start, end, suffix=""):
    return KdfFile(
        f.path,
        f.features[start:end].copy(),
        f.kinematics[start:end].copy(),
        None if f.nip_time is None else f.nip_time[start:end].copy(),
        f.trial_id + suffix,
        f.source_indices[start:end].copy(),
        f.feature_indices[start:end].copy(),
        f.nominal_interval,
        f.timestamp_info.copy(),
        None if f.original_nip_time is None else f.original_nip_time[start:end].copy(),
    )


def continuous_segments(f):
    if (
        f.nip_time is None
        or not np.isfinite(f.nip_time).all()
        or f.nominal_interval is None
    ):
        raise ValueError(
            f"{f.path}: Mamba preprocessing requires finite timestamps and a positive nominal interval"
        )
    dt = np.diff(np.asarray(f.nip_time, dtype=np.float64))
    cuts = np.r_[
        0,
        np.flatnonzero(
            (dt <= 0)
            | (dt > 1.5 * f.nominal_interval)
            | (np.diff(f.source_indices) != 1)
        )
        + 1,
        len(f.features),
    ]
    return [
        slice_file(f, int(a), int(b), f"_continuous{a}")
        for a, b in zip(cuts[:-1], cuts[1:])
    ]


def pair(f, lag):
    start, end = max(0, lag), min(len(f.features), len(f.features) + lag)
    return start, max(start, end)


def alignment_score(files, lag):
    cross = z2 = y2 = None
    count = 0
    for f in files:
        a, b = pair(f, lag)
        if b <= a:
            continue
        z = f.features[a - lag : b - lag].astype(np.float64)
        y = f.kinematics[a:b].astype(np.float64)
        y = np.concatenate((np.maximum(y, 0), np.minimum(y, 0)), axis=1)
        c, zz, yy = y.T @ z, (z * z).sum(0), (y * y).sum(0)
        cross = c if cross is None else cross + c
        z2 = zz if z2 is None else z2 + zz
        y2 = yy if y2 is None else y2 + yy
        count += b - a
    if count < 4:
        return None
    den = np.sqrt(y2[:, None] * z2[None, :])
    valid = den > 0
    return float(np.mean(np.abs(cross[valid] / den[valid]))) if valid.any() else None


def run(state, config):
    cfg = config.get("alignment", {})
    lo, hi = int(cfg.get("lag_min", -20)), int(cfg.get("lag_max", 20))
    if lo > hi:
        raise ValueError("alignment.lag_min must not exceed lag_max")
    for split in ("train", "val", "test"):
        setattr(
            state,
            split + "_files",
            [
                s
                for f in getattr(state, split + "_files")
                for s in continuous_segments(f)
            ],
        )
    scores = {
        str(lag): alignment_score(state.train_files, lag) for lag in range(lo, hi + 1)
    }
    if "lag" in cfg:
        lag = int(cfg["lag"])
    else:
        eligible = [l for l in range(lo, hi + 1) if scores[str(l)] is not None]
        if not eligible:
            raise ValueError("No usable training samples for lag estimation")
        lag = max(eligible, key=lambda l: scores[str(l)])
    excluded = {}
    for split in ("train", "val", "test"):
        files = []
        removed = 0
        for f in getattr(state, split + "_files"):
            a, b = pair(f, lag)
            removed += len(f.features) - (b - a)
            if b <= a:
                continue
            g = slice_file(f, a, b)
            g.features = f.features[a - lag : b - lag].copy()
            g.feature_indices = f.feature_indices[a - lag : b - lag].copy()
            files.append(g)
        setattr(state, split + "_files", files)
        excluded[split] = removed
    state.fitted_preprocessing["alignment"] = {
        "lag": lag,
        "scores": scores,
        "train_only": True,
        "excluded_samples": excluded,
        "effective_target_offset": lag
        + config.get("model", {}).get("prediction_lag_samples", 0),
    }
    return state

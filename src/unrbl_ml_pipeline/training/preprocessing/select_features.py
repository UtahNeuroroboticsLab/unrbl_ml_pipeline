"""Training-only reference Gram-Schmidt selection of original feature columns."""

import numpy as np


def candidates(value, count):
    if isinstance(value, str):
        if value in ("emg", "all"):
            return (
                np.arange(192, 720)
                if value == "emg" and count >= 720
                else np.arange(count)
            )
        if ":" in value:
            a, b = map(int, value.split(":"))
            value = np.arange(a, b + 1)
        else:
            value = [int(x) for x in value.split(",")]
    indices = np.asarray(value, dtype=int)
    if (
        indices.ndim != 1
        or not len(indices)
        or len(np.unique(indices)) != len(indices)
        or indices.min() < 0
        or indices.max() >= count
    ):
        raise ValueError("Feature indices must be unique in-range raw columns")
    return indices


def gram_schmidt(features, targets, indices, count=48, threshold=0.3):
    z, y = features.astype(np.float64).T, targets.astype(np.float64).T
    z = z - z.mean(1, keepdims=True)
    sep = []
    for row in y:
        if (row > 0).any():
            sep.append(np.maximum(row, 0))
        if (row < 0).any():
            sep.append(np.maximum(-row, 0))
    sep = np.asarray(sep) if sep else y.copy()
    sep -= sep.mean(1, keepdims=True)
    corr = np.abs(
        z[indices]
        @ sep.T
        / (
            np.maximum(np.linalg.norm(z[indices], axis=1), 1e-10)[:, None]
            * np.maximum(np.linalg.norm(sep, axis=1), 1e-10)[None]
        )
    )
    corr[corr < threshold] = 0
    order = []
    for col in range(corr.shape[1]):
        for i in np.argsort(-corr[:, col], kind="stable"):
            if corr[i, col] > 0 and indices[i] not in order:
                order.append(int(indices[i]))
    if not order:
        order = indices[np.argsort(-corr.max(1), kind="stable")].tolist()
    zz = z[order].copy()
    zz /= np.maximum(np.linalg.norm(zz, axis=1, keepdims=True), 1e-12)
    yy = y - y.mean(1, keepdims=True)
    selected, scores = [], []
    for _ in range(min(count, len(order))):
        residual = yy / np.maximum(np.linalg.norm(yy, axis=1, keepdims=True), 1e-12)
        correlations = np.abs(zz @ residual.T)
        ranked = np.argsort(-correlations.T.reshape(-1), kind="stable")
        chosen = next(
            int(i % len(order)) for i in ranked if int(i % len(order)) not in selected
        )
        scores.append(float(correlations[chosen].max()))
        selected.append(chosen)
        v = zz[chosen].copy()
        den = max(float(v @ v), 1e-12)
        yy -= np.outer(yy @ v / den, v)
        zz -= np.outer(zz @ v / den, v)
        zz /= np.maximum(np.linalg.norm(zz, axis=1, keepdims=True), 1e-12)
        zz[chosen] = 0
    return sorted(order[i] for i in selected), scores


def run(state, config):
    cfg = config.get("feature_selection", {})
    if not state.train_files:
        raise ValueError("Channel selection requires training data")
    n = state.train_files[0].features.shape[1]
    idx = candidates(cfg.get("candidates", "emg"), n)
    count = int(cfg.get("count", 48))
    if count < 1:
        raise ValueError("feature_selection.count must be positive")
    if "indices" in cfg:
        chosen, scores = sorted(candidates(cfg["indices"], n).tolist()), []
    else:
        chosen, scores = gram_schmidt(
            np.concatenate([f.features for f in state.train_files]),
            np.concatenate([f.kinematics for f in state.train_files]),
            idx,
            count,
            float(cfg.get("corr_threshold", 0.3)),
        )
    # Keep raw arrays: the serialized model performs selection exactly once.
    state.fitted_preprocessing["feature_selection"] = {
        "indices": chosen,
        "scores": scores,
        "candidate_indices": idx.tolist(),
        "count": len(chosen),
        "train_only": True,
        "embedded": True,
    }
    return state

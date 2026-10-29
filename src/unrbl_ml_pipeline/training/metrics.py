"""Regression metrics in original target units, with explicit undefined values."""
from __future__ import annotations

import numpy as np


def regression_metrics(target, prediction, names=None):
    """Aggregate and per-DOF errors, R-squared, and Pearson correlation.

    Use the TrueFaith report's degeneracy thresholds. Undefined R-squared or
    correlation is None, so saved JSON remains valid instead of containing NaN.
    """
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 2 or not len(target):
        raise ValueError("Metrics require nonempty matching [samples, DOFs] arrays")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Cannot report metrics for non-finite targets or predictions")
    error = prediction - target
    mse = np.mean(error ** 2, axis=0)
    mae = np.mean(np.abs(error), axis=0)
    names = names or [f"dof_{i}" for i in range(target.shape[1])]
    if len(names) != target.shape[1]:
        raise ValueError("DOF names must match the target dimension")
    per_dof = []
    for i, name in enumerate(names):
        truth_centered = target[:, i] - target[:, i].mean()
        pred_centered = prediction[:, i] - prediction[:, i].mean()
        ss_total = float(truth_centered @ truth_centered)
        ss_pred = float(pred_centered @ pred_centered)
        r2 = float(1 - np.sum(error[:, i] ** 2) / ss_total) if ss_total > 1e-8 else None
        corr = None
        if target[:, i].std() > 1e-8 and prediction[:, i].std() > 1e-8:
            corr = float(np.clip((truth_centered @ pred_centered) / np.sqrt(ss_total * ss_pred), -1, 1))
        per_dof.append({"name": name, "mse": float(mse[i]), "rmse": float(np.sqrt(mse[i])),
                        "mae": float(mae[i]), "r2": r2, "corr": corr})
    correlations = [row["corr"] for row in per_dof if row["corr"] is not None]
    return {
        "samples": len(target),
        "aggregate": {
            "mse": float(mse.mean()), "rmse": float(np.sqrt(mse.mean())),
            "mae": float(mae.mean()), "mean_dof_rmse": float(np.sqrt(mse).mean()),
            "mean_correlation": float(np.mean(correlations)) if correlations else None,
        },
        "per_dof": per_dof,
    }

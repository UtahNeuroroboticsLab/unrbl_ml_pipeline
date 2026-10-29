"""
Preprocessing step:
filter loaded KDF files by correlation quality score.

Author: Leonardo M. Ferrisi
Date: 2026-10-29
"""
from __future__ import annotations

import os

import numpy as np


def correlation_quality_score(features, kinematics):
    """Per-file data quality: mean best |Pearson r| across kinematic DOFs.

    Returns ``(score, per_dof_best_r)``.
    """
    n_dofs = kinematics.shape[1]
    best_r = np.zeros(n_dofs)
    feat_std = features.std(axis=0)
    valid_feat_cols = np.where(feat_std > 1e-8)[0]
    for d in range(n_dofs):
        y = kinematics[:, d]
        if y.std() < 1e-8 or valid_feat_cols.size == 0:
            best_r[d] = 0.0
            continue
        x = features[:, valid_feat_cols]
        x_c = x - x.mean(axis=0, keepdims=True)
        y_c = y - y.mean()
        num = x_c.T @ y_c
        denom = np.sqrt((x_c ** 2).sum(axis=0)) * np.sqrt((y_c ** 2).sum())
        denom[denom < 1e-12] = np.nan
        r = num / denom
        best_r[d] = min(1.0, np.nanmax(np.abs(r))) if np.isfinite(r).any() else 0.0
    return float(np.mean(best_r)), best_r


def run(state, config):
    """Keep only files whose quality score >= ``data.beta_threshold``.

    Config keys used:
        data.beta_threshold  — minimum score (default 0.0, i.e. keep all)
    """
    beta = float(config.get("data", {}).get("beta_threshold", 0.0))
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("data.beta_threshold must be between 0 and 1")
    records = {r["path"]: r for r in state.selection_report}
    kept = []
    for f in state.files:
        score, per_dof = correlation_quality_score(f.features, f.kinematics)
        record = records.get(os.path.abspath(f.path))
        if record is not None:
            record.update(score=score, per_dof_best_r=per_dof.tolist(),
                          status="kept" if score >= beta else "rejected",
                          reason=None if score >= beta else f"Score below beta {beta}")
        if score >= beta:
            kept.append(f)
            if state.progress is not None:
                state.progress.file(f.path, config, decision="KEEP", score=score,
                                    samples=len(f.features))
        else:
            state.rejected.append(f)
            if state.progress is not None:
                state.progress.file(f.path, config, decision="REJECT", score=score, threshold=beta)
    state.files = kept
    if not kept:
        raise RuntimeError(
            "Every candidate .kdf file was rejected by the quality filter "
            f"(beta_threshold={beta:.3f}). Lower data.beta_threshold or inspect data."
        )
    return state

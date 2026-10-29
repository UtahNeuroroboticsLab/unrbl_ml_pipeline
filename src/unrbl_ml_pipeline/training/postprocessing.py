"""Post-processing filters applied to decoder output (predicted kinematics).

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Ported from TrueFaith CLSTM_Decoder_Pipeline and CLSTM_Kalman_Pipeline.
All functions operate on numpy arrays and have no TensorFlow dependency.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Latching filter — causal quadratic-gain smoother
# ---------------------------------------------------------------------------

def latch_step(xhat, xhat_prev, LF_C=1.0):
    """One tick of the latching filter recurrence."""
    xhat = np.asarray(xhat, dtype=np.float64)
    xhat_prev = np.asarray(xhat_prev, dtype=np.float64)
    change = LF_C * np.square(xhat - xhat_prev)
    change = np.minimum(change, 1.0)
    xhat_out = xhat_prev + change * (xhat - xhat_prev)
    xhat_out = np.clip(xhat_out, -1.0, 1.0)
    return xhat_out


def apply_latching_filter(Y_pred, LF_C=1.0):
    """Apply latching filter to one contiguous trial. ``Y_pred``: ``[T, D]``."""
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    out = np.empty_like(Y_pred)
    xhat_prev = np.zeros(Y_pred.shape[1], dtype=np.float64)
    for t in range(Y_pred.shape[0]):
        xhat_prev = latch_step(Y_pred[t], xhat_prev, LF_C=LF_C)
        out[t] = xhat_prev
    return out.astype(Y_pred.dtype)


def apply_latching_filter_by_trial(Y_pred, trial_id, LF_C=1.0):
    """Per-trial latching filter; state resets at each new ``trial_id``."""
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    out = np.empty_like(Y_pred)
    n = len(trial_id)
    start = 0
    while start < n:
        end = start
        while end < n and trial_id[end] == trial_id[start]:
            end += 1
        out[start:end] = apply_latching_filter(Y_pred[start:end], LF_C=LF_C)
        start = end
    return out.astype(Y_pred.dtype)


# ---------------------------------------------------------------------------
# Deadband — stateless magnitude threshold
# ---------------------------------------------------------------------------

def apply_deadband(Y_pred, threshold=0.03):
    """Zero out predictions whose absolute value is below *threshold*."""
    Y_pred = np.asarray(Y_pred)
    return np.where(np.abs(Y_pred) < threshold, 0.0, Y_pred)


# ---------------------------------------------------------------------------
# Kalman filter — constant-velocity, 2-state per DOF
# ---------------------------------------------------------------------------

def kalman_step(z, x_prev, P_prev, Q, R):
    """One predict-update tick of the constant-velocity Kalman filter.

    Parameters
    ----------
    z : array [D]
        Measurement (raw decoder output).
    x_prev : array [D, 2]
        Previous state ``[position, velocity]`` per DOF.
    P_prev : array [D, 2, 2]
        Previous covariance per DOF.
    Q, R : array [D]
        Per-DOF process noise and measurement noise scalars.
    """
    x_pred = np.empty_like(x_prev)
    x_pred[:, 0] = x_prev[:, 0] + x_prev[:, 1]
    x_pred[:, 1] = x_prev[:, 1]

    P_pred = np.empty_like(P_prev)
    P_pred[:, 0, 0] = (
        P_prev[:, 0, 0] + P_prev[:, 0, 1] + P_prev[:, 1, 0] + P_prev[:, 1, 1] + Q
    )
    P_pred[:, 0, 1] = P_prev[:, 0, 1] + P_prev[:, 1, 1]
    P_pred[:, 1, 0] = P_prev[:, 1, 0] + P_prev[:, 1, 1]
    P_pred[:, 1, 1] = P_prev[:, 1, 1] + Q

    S = P_pred[:, 0, 0] + R
    K = np.stack([P_pred[:, 0, 0] / S, P_pred[:, 1, 0] / S], axis=-1)

    innovation = z - x_pred[:, 0]
    x_out = x_pred + K * innovation[:, None]

    P_out = P_pred.copy()
    P_out[:, 0, 0] = (1 - K[:, 0]) * P_pred[:, 0, 0]
    P_out[:, 0, 1] = (1 - K[:, 0]) * P_pred[:, 0, 1]
    P_out[:, 1, 0] = P_pred[:, 1, 0] - K[:, 1] * P_pred[:, 0, 0]
    P_out[:, 1, 1] = P_pred[:, 1, 1] - K[:, 1] * P_pred[:, 0, 1]

    return x_out, P_out


def apply_kalman_filter(Y_pred, process_noise=0.01, measurement_noise=0.05):
    """Constant-velocity Kalman filter over one contiguous trial.

    ``Y_pred``: ``[T, D]``. Returns smoothed position estimates, same shape.
    """
    orig_dtype = np.asarray(Y_pred).dtype
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    T, d = Y_pred.shape
    Q = np.full(d, process_noise, dtype=np.float64)
    R = np.full(d, measurement_noise, dtype=np.float64)

    x = np.zeros((d, 2), dtype=np.float64)
    x[:, 0] = Y_pred[0]
    P = np.zeros((d, 2, 2), dtype=np.float64)
    P[:, 0, 0] = 1.0
    P[:, 1, 1] = 1.0

    out = np.empty_like(Y_pred)
    out[0] = Y_pred[0]
    for t in range(1, T):
        x, P = kalman_step(Y_pred[t], x, P, Q, R)
        out[t] = x[:, 0]
    return out.astype(orig_dtype)


def apply_kalman_filter_by_trial(
    Y_pred, trial_id, process_noise=0.01, measurement_noise=0.05
):
    """Per-trial Kalman filter; state resets at each new ``trial_id``."""
    orig_dtype = np.asarray(Y_pred).dtype
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    out = np.empty_like(Y_pred)
    n = len(trial_id)
    start = 0
    while start < n:
        end = start
        while end < n and trial_id[end] == trial_id[start]:
            end += 1
        out[start:end] = apply_kalman_filter(
            Y_pred[start:end],
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
        start = end
    return out.astype(orig_dtype)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_postprocessing(Y_pred, trial_id, config):
    """Apply all enabled post-processing steps in order.

    *config* is the ``"postprocessing"`` dict from the model config, e.g.::

        {
            "latching_filter": {"enabled": True, "LF_C": 0.2},
            "deadband": {"enabled": True, "threshold": 0.04},
            "kalman_filter": {"enabled": True, "process_noise": 0.01,
                              "measurement_noise": 0.05},
        }

    Steps run in a fixed order: latching filter -> deadband -> Kalman filter.
    """
    if config is None:
        return Y_pred

    lf_cfg = config.get("latching_filter", {})
    if lf_cfg.get("enabled", False):
        Y_pred = apply_latching_filter_by_trial(
            Y_pred, trial_id, LF_C=lf_cfg.get("LF_C", 1.0)
        )

    db_cfg = config.get("deadband", {})
    if db_cfg.get("enabled", False):
        Y_pred = apply_deadband(Y_pred, threshold=db_cfg.get("threshold", 0.03))

    kf_cfg = config.get("kalman_filter", {})
    if kf_cfg.get("enabled", False):
        Y_pred = apply_kalman_filter_by_trial(
            Y_pred,
            trial_id,
            process_noise=kf_cfg.get("process_noise", 0.01),
            measurement_noise=kf_cfg.get("measurement_noise", 0.05),
        )

    return Y_pred

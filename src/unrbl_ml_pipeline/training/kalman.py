"""Kalman filter decoder: OLS-fit linear-Gaussian model with steady-state gain.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Ported from TrueFaith/KF_Decoder_Pipeline/lib/kf_model.py.

State transition:   x_t = A x_{t-1} + w_t,   w_t ~ N(0, W)
Observation model:  z_t = C x_t     + v_t,   v_t ~ N(0, Q)

  x = kinematics (what we want to estimate)
  z = features   (what we actually measure)

A and C are fit by ordinary least squares; W and Q are residual covariances.
The steady-state Kalman gain is solved once via the discrete algebraic
Riccati equation, not recomputed every timestep.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

from .estimator import Estimator


def _steady_state_gain(A, C, W, Q):
    """Steady-state Kalman gain via the discrete algebraic Riccati equation.

    Falls back to iterated Riccati recursion if the direct DARE solve fails
    (e.g. near-singular Q from highly collinear EMG features).
    """
    try:
        from scipy.linalg import solve_discrete_are
        P = solve_discrete_are(A.T, C.T, W, Q)
    except (ImportError, ValueError, np.linalg.LinAlgError):
        n = A.shape[0]
        P = np.eye(n)
        for _ in range(500):
            P_pred = A @ P @ A.T + W
            S = C @ P_pred @ C.T + Q
            K = P_pred @ C.T @ np.linalg.pinv(S)
            P_new = (np.eye(n) - K @ C) @ P_pred
            if np.allclose(P_new, P, atol=1e-10):
                P = P_new
                break
            P = P_new
        return P_pred @ C.T @ np.linalg.pinv(C @ P_pred @ C.T + Q)

    S = C @ P @ C.T + Q
    K = P @ C.T @ np.linalg.pinv(S)
    return K


@dataclass
class KalmanFilterDecoder(Estimator):
    """Linear-Gaussian Kalman filter decoder.

    Use the ``fit`` or ``fit_multi_file`` classmethods to construct a fitted
    instance; do not instantiate directly unless you already have the matrices.
    """

    A: np.ndarray
    C: np.ndarray
    W: np.ndarray
    Q: np.ndarray
    K: np.ndarray
    x_mean: np.ndarray

    @classmethod
    def fit(cls, X, Y):
        """Fit from pooled feature/kinematic arrays.

        Parameters
        ----------
        X : ndarray ``[T, n_features]``
            Features (observations).
        Y : ndarray ``[T, n_dofs]``
            Kinematics (states).

        Returns
        -------
        KalmanFilterDecoder
            Fitted instance.
        """
        features = np.asarray(X, dtype=np.float64)
        kinematics = np.asarray(Y, dtype=np.float64)

        x_t = kinematics[1:, :]
        x_tm1 = kinematics[:-1, :]
        A, _, _, _ = np.linalg.lstsq(x_tm1, x_t, rcond=None)
        A = A.T
        resid_state = x_t.T - A @ x_tm1.T
        if resid_state.shape[1] > 1:
            W = np.cov(resid_state)
        else:
            W = np.eye(A.shape[0]) * 1e-6

        C, _, _, _ = np.linalg.lstsq(kinematics, features, rcond=None)
        C = C.T
        resid_obs = features.T - C @ kinematics.T
        if resid_obs.shape[1] > 1:
            Q = np.cov(resid_obs)
        else:
            Q = np.eye(C.shape[0]) * 1e-6

        W = W + np.eye(W.shape[0]) * 1e-8
        Q = Q + np.eye(Q.shape[0]) * 1e-6

        K = _steady_state_gain(A, C, W, Q)
        return cls(
            A=A, C=C, W=W, Q=Q, K=K, x_mean=kinematics.mean(axis=0),
        )

    @classmethod
    def fit_multi_file(cls, files):
        """Fit A per-file (weighted average) to avoid cross-file boundary artefacts.

        Parameters
        ----------
        files : list
            Objects with ``.features`` ``[T, F]`` and ``.kinematics`` ``[T, K]``
            attributes.

        Returns
        -------
        KalmanFilterDecoder
        """
        n_dofs = files[0].kinematics.shape[1]
        A_sum = np.zeros((n_dofs, n_dofs), dtype=np.float64)
        W_resid_chunks = []
        total_transitions = 0

        for f in files:
            kin = np.asarray(f.kinematics, dtype=np.float64)
            x_t = kin[1:, :]
            x_tm1 = kin[:-1, :]
            if x_tm1.shape[0] < 2:
                continue
            A_i, _, _, _ = np.linalg.lstsq(x_tm1, x_t, rcond=None)
            A_i = A_i.T
            n_i = x_tm1.shape[0]
            A_sum += A_i * n_i
            total_transitions += n_i
            W_resid_chunks.append(x_t.T - A_i @ x_tm1.T)

        if total_transitions == 0:
            raise ValueError("Kalman training needs a segment with at least three samples")
        A = A_sum / total_transitions
        resid_state = np.concatenate(W_resid_chunks, axis=1)
        W = np.cov(resid_state) + np.eye(n_dofs) * 1e-8

        all_features = np.concatenate(
            [np.asarray(f.features, dtype=np.float64) for f in files], axis=0,
        )
        all_kinematics = np.concatenate(
            [np.asarray(f.kinematics, dtype=np.float64) for f in files], axis=0,
        )
        C, _, _, _ = np.linalg.lstsq(all_kinematics, all_features, rcond=None)
        C = C.T
        resid_obs = all_features.T - C @ all_kinematics.T
        Q = np.cov(resid_obs) + np.eye(C.shape[0]) * 1e-6

        K = _steady_state_gain(A, C, W, Q)
        return cls(
            A=A, C=C, W=W, Q=Q, K=K,
            x_mean=all_kinematics.mean(axis=0),
        )

    def predict(self, X):
        """Run inference on ``[T, n_features]`` (delegates to ``predict_sequence``)."""
        return self.predict_sequence(np.asarray(X, dtype=np.float64))

    def predict_sequence(self, features):
        """Forward Kalman recursion over ``[T, n_features]``, returns ``[T, n_dofs]``."""
        features = np.asarray(features, dtype=np.float64)
        T = features.shape[0]
        n_dofs = self.A.shape[0]
        x_est = np.zeros((T, n_dofs), dtype=np.float64)
        x_prev = self.x_mean.copy()
        for t in range(T):
            x_pred = self.A @ x_prev
            innovation = features[t, :] - self.C @ x_pred
            x_prev = x_pred + self.K @ innovation
            x_est[t, :] = x_prev
        return x_est

    def save(self, path):
        """Persist the fitted decoder to *path* as a pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Load a previously saved decoder from *path*."""
        with open(path, "rb") as f:
            return pickle.load(f)

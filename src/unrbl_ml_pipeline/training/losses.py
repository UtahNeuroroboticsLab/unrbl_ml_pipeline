"""Shared loss functions for all model pipelines.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Ported from TrueFaith/CLSTM_Decoder_Pipeline/lib/losses.py.  Every loss
is a plain function or a factory that returns one.  ``get_loss(config)``
is the single entry point the training loop uses for non-trajectory
losses; trajectory losses are built separately via
``make_predicted_trajectory_loss_fast()``.
"""
from __future__ import annotations

import tensorflow as tf

# ---------------------------------------------------------------------------
# Simple losses
# ---------------------------------------------------------------------------

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def huber_loss(delta=1.0):
    h = tf.keras.losses.Huber(delta=delta)

    def _loss(y_true, y_pred):
        return h(y_true, y_pred)

    return _loss


# ---------------------------------------------------------------------------
# Weighted / packed-target losses
# ---------------------------------------------------------------------------

def mse_with_delta_penalty(delta_weight=0.1):
    """MSE + frame-to-frame delta penalty.

    Requires packed targets ``[batch, 2, num_dofs]``: slot 0 is the
    current target, slot 1 is the previous-timestep target.
    """
    def _loss(y_true, y_pred):
        y_cur = y_true[:, 0, :]
        y_prev = y_true[:, 1, :]
        mse_cur = tf.reduce_mean(tf.square(y_cur - y_pred))
        true_delta = y_cur - y_prev
        pred_delta = y_pred - y_prev
        delta_mse = tf.reduce_mean(tf.square(true_delta - pred_delta))
        return mse_cur + delta_weight * delta_mse

    return _loss


def idle_weighted_mse(idle_threshold=0.05, idle_weight=5.0):
    """MSE that upweights idle-DOF errors (``|y_true| < threshold``)."""
    idle_threshold = float(idle_threshold)
    idle_weight = float(idle_weight)

    def _loss(y_true, y_pred):
        sq_err = tf.square(y_true - y_pred)
        is_idle = tf.abs(y_true) < idle_threshold
        weight = tf.where(
            is_idle,
            tf.constant(idle_weight, sq_err.dtype),
            tf.constant(1.0, sq_err.dtype),
        )
        return tf.reduce_mean(weight * sq_err)

    return _loss


def recency_weighted_mse(min_weight=0.1):
    """MSE scaled by a per-window recency weight smuggled into packed targets.

    Requires packed targets ``[batch, 2, num_dofs]``: slot 0 is the
    regression target, slot 1 is the recency weight (broadcast across DOFs).
    """
    min_weight = float(min_weight)

    def _loss(y_true, y_pred):
        y_cur = y_true[:, 0, :]
        weight = tf.maximum(y_true[:, 1, :], min_weight)
        sq_err = tf.square(y_cur - y_pred)
        return tf.reduce_mean(weight * sq_err)

    return _loss


# ---------------------------------------------------------------------------
# Metrics for packed-target losses
# ---------------------------------------------------------------------------

def packed_current_mae(y_true, y_pred):
    """MAE against the current-timestep target only (unpacks slot 0)."""
    y_cur = y_true[:, 0, :] if len(y_true.shape) == 3 else y_true
    return tf.reduce_mean(tf.abs(y_cur - y_pred))


def packed_current_mse(y_true, y_pred):
    """MSE against the current-timestep target only (unpacks slot 0)."""
    y_cur = y_true[:, 0, :] if len(y_true.shape) == 3 else y_true
    return tf.reduce_mean(tf.square(y_cur - y_pred))


# ---------------------------------------------------------------------------
# Trajectory losses
# ---------------------------------------------------------------------------

TRAJECTORY_LOSS_TYPES = ("predicted_trajectory_loss_fast",)


def is_trajectory_loss(loss_type):
    """True for losses that need the trajectory training branch."""
    return str(loss_type).lower() in TRAJECTORY_LOSS_TYPES


def _trajectory_derivative_penalty(error, mode="mse", huber_delta=0.05):
    mode = str(mode).lower()
    if mode in ("mse", "mean_squared_error"):
        return tf.reduce_mean(tf.square(error))
    if mode == "huber":
        abs_err = tf.abs(error)
        delta = float(huber_delta)
        quadratic = tf.minimum(abs_err, delta)
        linear = abs_err - quadratic
        return tf.reduce_mean(0.5 * tf.square(quadratic) + delta * linear)
    raise ValueError(
        f"Unknown loss.derivative_penalty '{mode}'; use 'mse' or 'huber'."
    )


def make_predicted_trajectory_loss_fast(
    velocity_lambda=0.0,
    acceleration_lambda=0.05,
    jerk_lambda=0.8,
    high_jerk_lambda=10.0,
    high_pred_jerk_threshold=0.25,
    true_jerk_trust_threshold=0.20,
    derivative_penalty="mse",
    derivative_huber_delta=0.05,
):
    """Position loss + derivative matching over consecutive model predictions.

    Inputs ``y_true`` and ``y_pred`` are both
    ``[batch, trajectory_length, num_dofs]``.
    """
    velocity_lambda = float(velocity_lambda)
    acceleration_lambda = float(acceleration_lambda)
    jerk_lambda = float(jerk_lambda)
    high_jerk_lambda = float(high_jerk_lambda)
    high_pred_jerk_threshold = float(high_pred_jerk_threshold)
    true_jerk_trust_threshold = float(true_jerk_trust_threshold)
    derivative_penalty = str(derivative_penalty).lower()
    derivative_huber_delta = float(derivative_huber_delta)

    def loss_fn(y_true, y_pred):
        total = tf.reduce_mean(tf.square(y_true - y_pred))

        if velocity_lambda > 0.0:
            true_vel = y_true[:, 1:, :] - y_true[:, :-1, :]
            pred_vel = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            total = total + velocity_lambda * _trajectory_derivative_penalty(
                pred_vel - true_vel, derivative_penalty, derivative_huber_delta
            )

        if acceleration_lambda > 0.0:
            true_acc = (
                y_true[:, 2:, :] - 2.0 * y_true[:, 1:-1, :] + y_true[:, :-2, :]
            )
            pred_acc = (
                y_pred[:, 2:, :] - 2.0 * y_pred[:, 1:-1, :] + y_pred[:, :-2, :]
            )
            total = total + acceleration_lambda * _trajectory_derivative_penalty(
                pred_acc - true_acc, derivative_penalty, derivative_huber_delta
            )

        if jerk_lambda > 0.0 or high_jerk_lambda > 0.0:
            true_jerk = (
                y_true[:, 3:, :]
                - 3.0 * y_true[:, 2:-1, :]
                + 3.0 * y_true[:, 1:-2, :]
                - y_true[:, :-3, :]
            )
            pred_jerk = (
                y_pred[:, 3:, :]
                - 3.0 * y_pred[:, 2:-1, :]
                + 3.0 * y_pred[:, 1:-2, :]
                - y_pred[:, :-3, :]
            )
            jerk_error = pred_jerk - true_jerk
            trusted = tf.abs(true_jerk) <= tf.cast(
                true_jerk_trust_threshold, true_jerk.dtype
            )

            if derivative_penalty == "huber":
                abs_err = tf.abs(jerk_error)
                delta = tf.cast(derivative_huber_delta, jerk_error.dtype)
                quadratic = tf.minimum(abs_err, delta)
                linear = abs_err - quadratic
                normal_penalty = 0.5 * tf.square(quadratic) + delta * linear
            else:
                normal_penalty = tf.square(jerk_error)

            normal_penalty = tf.where(
                trusted, normal_penalty, tf.zeros_like(normal_penalty)
            )

            high_pred = tf.abs(pred_jerk) > tf.cast(
                high_pred_jerk_threshold, pred_jerk.dtype
            )
            high_penalty = tf.square(pred_jerk)

            jerk_penalty = tf.where(
                high_pred,
                tf.cast(high_jerk_lambda, pred_jerk.dtype) * high_penalty,
                tf.cast(jerk_lambda, pred_jerk.dtype) * normal_penalty,
            )
            total = total + tf.reduce_mean(jerk_penalty)

        return total

    loss_fn.__name__ = "predicted_trajectory_loss_fast"
    return loss_fn


# ---------------------------------------------------------------------------
# Trajectory metrics
# ---------------------------------------------------------------------------

def trajectory_acceleration_rmse(y_true, y_pred):
    true_acc = y_true[:, 2:, :] - 2.0 * y_true[:, 1:-1, :] + y_true[:, :-2, :]
    pred_acc = y_pred[:, 2:, :] - 2.0 * y_pred[:, 1:-1, :] + y_pred[:, :-2, :]
    return tf.sqrt(tf.reduce_mean(tf.square(pred_acc - true_acc)))


def trajectory_jerk_rmse(y_true, y_pred):
    true_jerk = (
        y_true[:, 3:, :]
        - 3.0 * y_true[:, 2:-1, :]
        + 3.0 * y_true[:, 1:-2, :]
        - y_true[:, :-3, :]
    )
    pred_jerk = (
        y_pred[:, 3:, :]
        - 3.0 * y_pred[:, 2:-1, :]
        + 3.0 * y_pred[:, 1:-2, :]
        - y_pred[:, :-3, :]
    )
    return tf.sqrt(tf.reduce_mean(tf.square(pred_jerk - true_jerk)))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_loss(loss_config):
    """Build a loss function from config.

    Returns ``(loss_fn, needs_packed_targets)`` where *needs_packed_targets*
    is True when ``y_true`` must be ``[batch, 2, num_dofs]`` (packed with
    auxiliary data in slot 1).
    """
    loss_type = loss_config.get("type", "mse")

    if loss_type == "mse":
        return mse_loss, False
    if loss_type == "huber":
        return huber_loss(loss_config.get("huber_delta", 1.0)), False
    if loss_type == "mse_with_delta_penalty":
        return mse_with_delta_penalty(loss_config.get("delta_weight", 0.1)), True
    if loss_type == "idle_weighted_mse":
        return idle_weighted_mse(
            loss_config.get("idle_threshold", 0.05),
            loss_config.get("idle_weight", 5.0),
        ), False
    if loss_type == "recency_weighted_mse":
        return recency_weighted_mse(
            loss_config.get("recency_min_weight", 0.1)
        ), True
    raise ValueError(
        f"Unknown loss.type '{loss_type}'. Available: mse, huber, "
        "mse_with_delta_penalty, idle_weighted_mse, recency_weighted_mse, "
        "predicted_trajectory_loss_fast (built via "
        "make_predicted_trajectory_loss_fast(), not get_loss())."
    )

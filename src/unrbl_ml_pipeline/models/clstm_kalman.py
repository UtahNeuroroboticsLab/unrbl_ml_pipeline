"""CLSTM decoder with Kalman filter post-processing.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Same CLSTM architecture as ``clstm_decoder`` but configured with
recency-weighted MSE loss and a Kalman filter post-processing step instead
of (or in addition to) the deadband filter.
"""

name = "clstm_kalman"
description = "CLSTM decoder with Kalman filter post-processing"
paper = "Ferrisi (in preparation)"
model_type = "keras"

default_config = {
    "preprocessing_steps": [
        "load_kdf",
        "quality_filter",
        "split_by_detected_trial_deadzone",
        "normalize",
        "window",
    ],
    "model": {"window_size": 10},
    "data": {"beta_threshold": 0.2},
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "epochs": 200,
    "batch_size": 128,
    "loss": {
        "type": "recency_weighted_mse",
        "recency_min_weight": 0.1,
    },
    "early_stopping_patience": 18,
    "reduce_lr_patience": 8,
    "architecture": {
        "conv_filters": 64,
        "conv_kernel_size": 8,
        "use_batch_norm": True,
        "lstm_units": 128,
        "dense_units": 50,
        "dropout": 0.0,
        "use_deadband_activation_output": True,
    },
    "postprocessing": {
        "deadband": {"enabled": True, "threshold": 0.04},
        "kalman_filter": {
            "enabled": True,
            "process_noise": 0.01,
            "measurement_noise": 0.05,
        },
    },
}


def build(input_shape, num_classes, norm_stats=None, **config):
    """Return an uncompiled CLSTM-Kalman model (delegates to clstm_decoder)."""
    from unrbl_ml_pipeline.models import clstm_decoder
    return clstm_decoder.build(
        input_shape, num_classes, norm_stats=norm_stats,
        stateful=False, **config,
    )

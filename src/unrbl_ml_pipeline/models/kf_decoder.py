"""Linear-Gaussian Kalman filter decoder (OLS fit, steady-state gain).

Author: Leonardo M. Ferrisi
Date: 2026-10-29

A standard linear-Gaussian Kalman filter in the neural decoding convention:
state = kinematics, observation = features.  A and C are fit by OLS on
training data; the steady-state Kalman gain is solved once via the DARE.

This is a non-neural-network model (``model_type = "estimator"``).
``build()`` returns the *class*, not an instance -- the training loop
instantiates it by calling ``KalmanFilterDecoder.fit()``.
"""

name = "kf_decoder"
description = "Linear-Gaussian Kalman filter decoder (OLS fit, steady-state gain)"
model_type = "estimator"

default_config = {
    "preprocessing_steps": [
        "load_kdf",
        "quality_filter",
        "split_by_file",
    ],
    "beta_threshold": 0.2,
    "postprocessing": {
        "deadband": {"enabled": True, "threshold": 0.04},
    },
}


def build(**config):
    """Return the KalmanFilterDecoder class (not an instance).

    The training loop should call ``KalmanFilterDecoder.fit(features, kinematics)``
    or ``KalmanFilterDecoder.fit_multi_file(files)`` to produce a fitted instance.
    """
    from unrbl_ml_pipeline.training.kalman import KalmanFilterDecoder
    return KalmanFilterDecoder

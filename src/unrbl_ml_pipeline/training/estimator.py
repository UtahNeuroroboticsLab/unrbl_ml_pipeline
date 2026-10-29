"""Estimator protocol for non-Keras models.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Any non-neural-network model (Kalman filter, sklearn wrapper, Wiener filter,
etc.) should implement this interface so the training loop can handle it
generically via ``model_type = "estimator"``.
"""
from __future__ import annotations


class Estimator:
    """Base class / protocol for non-Keras models.

    Subclass this and implement the four methods, or duck-type them on a
    standalone class -- the training loop checks for ``fit`` / ``predict``
    by attribute, not by ``isinstance``.
    """

    def fit(self, X, Y):
        """Fit the model on training data.

        Parameters
        ----------
        X : numpy array
            Features, shape depends on model.
        Y : numpy array
            Targets, shape depends on model.
        """
        raise NotImplementedError

    def predict(self, X):
        """Run inference on new data.

        Parameters
        ----------
        X : numpy array
            Features with the same column layout as ``fit()``.

        Returns
        -------
        numpy array
            Predicted targets.
        """
        raise NotImplementedError

    def save(self, path):
        """Persist the fitted model to *path*."""
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        """Load a previously saved model from *path*."""
        raise NotImplementedError

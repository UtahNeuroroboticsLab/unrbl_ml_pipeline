"""Integration tests for the training API.  Requires TF (3.10+ only).

Author: Leonardo M. Ferrisi
Date: 2026-10-29
"""

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="training requires Python 3.10+ (TensorFlow)",
)


@pytest.fixture()
def classification_data():
    rng = np.random.RandomState(42)
    x = rng.randn(60, 10).astype(np.float32)
    y = rng.randint(0, 3, size=60)
    return x, y


@pytest.fixture()
def regression_data():
    rng = np.random.RandomState(42)
    x = rng.randn(60, 10).astype(np.float32)
    y = rng.randn(60, 2).astype(np.float32)
    return x, y


def test_train_with_module(classification_data):
    from unrbl_ml_pipeline import models
    from unrbl_ml_pipeline.training import train

    model, history = train(
        models.ExampleModel,
        classification_data[0],
        classification_data[1],
        epochs=2,
        batch_size=16,
        validation_split=0.0,
    )
    assert model is not None
    assert hasattr(history, "history")
    assert model.output_shape[-1] == 3


def test_train_with_string(classification_data):
    from unrbl_ml_pipeline.training import train

    model, _history = train(
        "examplemodel",
        classification_data[0],
        classification_data[1],
        epochs=2,
        batch_size=16,
        validation_split=0.0,
    )
    assert model.output_shape[-1] == 3


def test_train_onehot_labels():
    from unrbl_ml_pipeline.training import train

    rng = np.random.RandomState(0)
    x = rng.randn(40, 8).astype(np.float32)
    y = np.zeros((40, 4), dtype=np.float32)
    for i in range(40):
        y[i, rng.randint(0, 4)] = 1.0

    model, _ = train("examplemodel", x, y, epochs=1, validation_split=0.0)
    assert model.output_shape[-1] == 4


def test_train_2d_input():
    from unrbl_ml_pipeline.training import train

    rng = np.random.RandomState(1)
    x = rng.randn(40, 128, 8).astype(np.float32)
    y = rng.randint(0, 5, size=40)

    model, _ = train("examplemodel", x, y, epochs=1, validation_split=0.0)
    assert model.output_shape[-1] == 5


def test_train_regression(regression_data):
    from unrbl_ml_pipeline.training import train

    model, history = train(
        "examplemodel",
        regression_data[0],
        regression_data[1],
        task="regression",
        epochs=2,
        validation_split=0.0,
    )
    assert model.output_shape[-1] == 2
    assert "mae" in history.history


def test_train_regression_1d_target():
    from unrbl_ml_pipeline.training import train

    rng = np.random.RandomState(2)
    x = rng.randn(40, 10).astype(np.float32)
    y = rng.randn(40).astype(np.float32)

    model, _ = train(
        "examplemodel", x, y, task="regression", epochs=1, validation_split=0.0
    )
    assert model.output_shape[-1] == 1


def test_train_custom_optimizer(classification_data):
    from unrbl_ml_pipeline.training import train

    model, _ = train(
        "examplemodel",
        classification_data[0],
        classification_data[1],
        optimizer="sgd",
        epochs=1,
        validation_split=0.0,
    )
    assert model is not None


def test_train_invalid_task(classification_data):
    from unrbl_ml_pipeline.training import train

    with pytest.raises(ValueError, match="task must be"):
        train(
            "examplemodel",
            classification_data[0],
            classification_data[1],
            task="invalid",
        )

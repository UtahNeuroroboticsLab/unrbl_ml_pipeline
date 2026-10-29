"""Tests for the model registry.  Must stay 3.7-syntax-clean.

Author: Leonardo M. Ferrisi
Date: 2026-10-29
"""

import pytest

from unrbl_ml_pipeline import models


def test_list_models_returns_sorted_strings():
    result = models.list_models()
    assert isinstance(result, list)
    assert len(result) > 0
    assert result == sorted(result)
    assert all(isinstance(n, str) for n in result)


def test_get_returns_module_with_build():
    mod = models.get("examplemodel")
    assert callable(mod.build)


def test_get_is_case_insensitive():
    assert models.get("ExampleModel") is models.get("examplemodel")


def test_get_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="not a registered model"):
        models.get("nonexistent_model_xyz")


def test_getattr_access():
    mod = models.ExampleModel
    assert callable(mod.build)


def test_getattr_unknown_raises_attributeerror():
    with pytest.raises(AttributeError):
        _ = models.nonexistent_model_xyz


def test_example_model_has_metadata():
    mod = models.get("examplemodel")
    assert isinstance(mod.name, str)
    assert isinstance(mod.description, str)

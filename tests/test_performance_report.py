"""Numeric report semantics and prediction/source alignment."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from unrbl_ml_pipeline.training.metrics import regression_metrics
from unrbl_ml_pipeline.training.reporting import test_source_segments as source_segments


def test_report_metrics_handle_constant_signals_and_negative_r2():
    truth = np.array([[0, 1, 0], [1, 1, 1], [2, 1, 2]], dtype=float)
    pred = np.array([[0, 0, 1], [2, 2, 1], [4, 4, 1]], dtype=float)
    result = regression_metrics(truth, pred, ["varying", "constant target", "constant prediction"])
    first, second, third = result["per_dof"]
    assert first["r2"] == pytest.approx(-1.5)
    assert first["corr"] == pytest.approx(1)
    assert second["r2"] is None and second["corr"] is None
    assert third["r2"] == pytest.approx(0) and third["corr"] is None
    assert result["aggregate"]["mean_correlation"] == pytest.approx(1)
    assert result["aggregate"]["mean_dof_rmse"] == pytest.approx(
        np.mean([row["rmse"] for row in result["per_dof"]]),
    )
    assert result["aggregate"]["mean_dof_rmse"] != pytest.approx(result["aggregate"]["rmse"])
    json.dumps(result, allow_nan=False)


def test_all_constant_dofs_produce_json_safe_undefined_values():
    result = regression_metrics(np.ones((5, 2)), np.ones((5, 2)))
    assert result["aggregate"]["rmse"] == 0
    assert result["aggregate"]["mean_correlation"] is None
    assert all(row["r2"] is None and row["corr"] is None for row in result["per_dof"])
    json.dumps(result, allow_nan=False)


def test_source_ranges_use_full_paths_and_window_lag(tmp_path):
    basename = "trainKDF_20260801-120000.kdf"
    files = [SimpleNamespace(path=tmp_path / folder / basename,
                             features=np.zeros((length, 2)), trial_id=folder)
             for folder, length in (("a", 8), ("b", 10))]
    results = {"config": {"preprocessing_steps": ["window"],
                           "model": {"window_size": 3, "prediction_lag_samples": 1}},
               "state": SimpleNamespace(test_files=files), "Y_test": np.zeros((12, 2))}
    segments = source_segments(results, "keras")
    assert [(s["start"], s["end"]) for s in segments] == [(0, 5), (5, 12)]
    assert segments[0]["path"] != segments[1]["path"]
    results["Y_test"] = np.zeros((18, 2))
    assert [(s["start"], s["end"]) for s in source_segments(results, "estimator")] == [(0, 8), (8, 18)]
    results["Y_test"] = np.zeros((17, 2))
    with pytest.raises(ValueError, match="source segments"):
        source_segments(results, "estimator")

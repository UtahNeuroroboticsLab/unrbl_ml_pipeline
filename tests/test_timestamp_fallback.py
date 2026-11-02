"""Opt-in sample-order timing, donor selection and boundary/report propagation."""

import os
from copy import deepcopy

import numpy as np
import pytest

from unrbl_ml_pipeline.cli import parse_training_args, training_overrides
from unrbl_ml_pipeline.training.preprocessing import run_pipeline
from unrbl_ml_pipeline.training.preprocessing._timing import apply_timing_policy
from unrbl_ml_pipeline.training.preprocessing._types import KdfFile, PipelineState
from unrbl_ml_pipeline.training.preprocessing.align_lag import continuous_segments


def write_kdf(path, times):
    rng = np.random.default_rng(72)
    x = rng.normal(size=(len(times), 5)).astype("float32")
    y = np.tanh(x[:, :2])
    with path.open("wb") as stream:
        np.asarray([1, 5, 2], "<f4").tofile(stream)
        np.column_stack((times, x, y)).astype("<f4").tofile(stream)


def config(path, **extra):
    return {
        "data": {
            "base_dir": str(path),
            "kdf_format": "cob",
            "require_usable_timestamps": True,
            "kinematic_column_indices": [0, 1],
            "beta_threshold": 0,
            **extra,
        }
    }


def test_opt_in_preserves_clocks_and_estimates_after_selection(tmp_path):
    real = np.arange(24) * 992.0
    real[12:] += 10000  # Retain this observed gap even when fallback is enabled.
    good = tmp_path / "trainKDF_20261020-000000.kdf"
    bad = tmp_path / "trainKDF_20261021-000000.kdf"
    outside = tmp_path / "trainKDF_20261001-000000.kdf"
    write_kdf(good, real)
    write_kdf(bad, np.full(24, 2491635.0))
    write_kdf(outside, np.arange(24) * 50000.0)
    cfg = config(tmp_path, date_from="2026-10-20")
    strict = run_pipeline(["load_kdf"], cfg)
    assert len(strict.files) == 1
    rejected = next(r for r in strict.selection_report if r["path"] == str(bad))
    assert rejected["status"] == "excluded"
    assert "failed to parse" not in rejected["reason"]
    cfg["data"]["allow_missing_timestamps"] = True
    result = run_pipeline(["load_kdf", "quality_filter"], cfg)
    assert len(result.files) == 2
    a, b = result.files
    np.testing.assert_array_equal(a.nip_time, real)
    assert len(continuous_segments(a)) == 2
    np.testing.assert_array_equal(b.nip_time, np.arange(24) * 992.0)
    np.testing.assert_array_equal(b.original_nip_time, np.full(24, 2491635.0))
    assert b.nip_time.dtype == np.float64
    assert b.timestamp_info["source"] == "sample_order"
    estimate = result.fitted_preprocessing["timing"]
    assert estimate["proxy_interval"] == 992
    assert [r["path"] for r in estimate["donors"]] == [str(good)]


@pytest.mark.parametrize("kind", ["missing", "constant", "nan", "inf"])
def test_unusable_clocks_use_explicit_interval(kind):
    times = {
        "missing": None,
        "constant": np.ones(8),
        "nan": np.full(8, np.nan),
        "inf": np.full(8, np.inf),
    }[kind]
    file = KdfFile("recording", np.ones((8, 2)), np.ones((8, 1)), times)
    state = PipelineState(selection_report=[{"path": os.path.abspath(file.path)}])
    cfg = {"data": {"allow_missing_timestamps": True, "sample_interval": 990}}
    result = apply_timing_policy([file], state, cfg)
    np.testing.assert_array_equal(result[0].nip_time, np.arange(8) * 990.0)
    assert result[0].nominal_interval == 990
    assert result[0].timestamp_info["interval_source"] == "explicit"


def test_no_clock_donors_needs_interval(tmp_path):
    write_kdf(tmp_path / "recording.kdf", np.ones(24))
    with pytest.raises(ValueError, match="sample-interval"):
        run_pipeline(["load_kdf"], config(tmp_path, allow_missing_timestamps=True))
    result = run_pipeline(
        ["load_kdf"],
        config(tmp_path, allow_missing_timestamps=True, sample_interval=992),
    )
    assert len(result.files) == 1


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"), True])
def test_invalid_proxy_intervals(interval):
    with pytest.raises(ValueError, match="sample_interval"):
        apply_timing_policy(
            [],
            PipelineState(),
            {"data": {"allow_missing_timestamps": True, "sample_interval": interval}},
        )


def test_cli_override_and_default():
    assert "data" not in training_overrides(parse_training_args([]))
    result = training_overrides(
        parse_training_args(["--allow-missing-timestamps", "--sample-interval", "992"]),
        {"data": {"allow_missing_timestamps": False, "sample_interval": 500}},
    )
    assert result["data"] == {
        "allow_missing_timestamps": True,
        "sample_interval": 992.0,
    }


def test_proxy_training_report_and_no_cross_file_windows(tmp_path):
    from unrbl_ml_pipeline.training import run_training

    data = tmp_path / "data"
    data.mkdir()
    for i in range(8):
        write_kdf(
            data / f"trainKDF_202610{i + 10}-000000.kdf",
            np.ones(32) if i % 2 else np.arange(32) * 992.0,
        )
    cfg = config(data, allow_missing_timestamps=True)
    cfg.update(
        preprocessing_steps=[
            "load_kdf",
            "quality_filter",
            "split_by_file",
            "align_lag",
            "select_features",
            "window",
        ],
        alignment={"lag": 0, "lag_min": 0, "lag_max": 0},
        model={"window_size": 5, "prediction_lag_samples": 1},
        architecture={"mamba_width": 8, "state_size": 4, "head_units": 8},
        epochs=1,
        streams=2,
        tbptt_steps=8,
    )
    _, result = run_training(
        "mamba_decoder", config=cfg, output_dir=tmp_path / "out", verbose=0
    )
    assert len(result["state"].files) == 8
    report = result["performance_path"].read_text()
    assert "4 selected recordings use sample-order proxy timestamps" in report
    assert "992" in report
    for split in ("train", "val", "test"):
        for rec in result["state"].provenance[split]:
            assert rec["timing"]["source"] in ("recorded", "sample_order")
            assert rec["end"] - rec["start"] == 27
            np.testing.assert_array_equal(
                np.diff(rec["target_times"]), np.full(26, 992)
            )
    # Check metadata survives further slicing rather than losing its provenance.
    copy = deepcopy(result["state"].train_files[0])
    assert continuous_segments(copy)[0].timestamp_info == copy.timestamp_info

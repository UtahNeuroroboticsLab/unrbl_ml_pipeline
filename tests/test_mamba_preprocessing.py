"""Fitted transformations, input formats and metadata boundary isolation."""

from copy import deepcopy

import numpy as np
import pytest

from unrbl_ml_pipeline.training.preprocessing._types import KdfFile, PipelineState
from unrbl_ml_pipeline.training.preprocessing.align_lag import run as align
from unrbl_ml_pipeline.training.preprocessing.load_kdf import read_kdf
from unrbl_ml_pipeline.training.preprocessing.select_features import run as select
from unrbl_ml_pipeline.training.preprocessing.window import run as window


def fixture_state():
    rng = np.random.default_rng(4)

    def file(name):
        x = rng.normal(size=(70, 5)).astype(np.float32)
        y = np.zeros((70, 2), np.float32)
        y[2:] = x[:-2, :2]
        return KdfFile(name, x, y, np.arange(70) * 1000)

    return PipelineState(
        train_files=[file("train")], val_files=[file("val")], test_files=[file("test")]
    )


def test_fits_exclude_heldout():
    a = fixture_state()
    b = deepcopy(a)
    b.val_files[0].features *= 100
    b.test_files[0].kinematics *= -100
    cfg = {
        "alignment": {"lag_min": -3, "lag_max": 3},
        "feature_selection": {"count": 2},
    }
    for state in (a, b):
        align(state, cfg)
        select(state, cfg)
    assert a.fitted_preprocessing == b.fitted_preprocessing
    assert a.fitted_preprocessing["alignment"]["lag"] == 2
    assert a.fitted_preprocessing["feature_selection"]["indices"] == [0, 1]


@pytest.mark.parametrize("lag", [-3, 0, 3])
def test_lag_and_source_mapping(lag):
    s = fixture_state()
    original = s.train_files[0]
    cfg = {
        "alignment": {"lag": lag, "lag_min": 0, "lag_max": 0},
        "model": {"window_size": 5, "prediction_lag_samples": 1},
    }
    align(s, cfg)
    window(s, cfg)
    r = s.provenance["train"][0]
    np.testing.assert_array_equal(r["target_indices"] - r["input_end_indices"], lag + 1)
    np.testing.assert_array_equal(s.Y_train, original.kinematics[r["target_indices"]])
    np.testing.assert_array_equal(
        s.X_train[:, -1], original.features[r["input_end_indices"]]
    )
    assert len(s.train_files[0].features) == 70 - abs(lag)


def test_timestamp_and_split_gaps():
    s = fixture_state()
    s.train_files[0].nip_time[30:] += 4000
    cfg = {
        "alignment": {"lag": 0, "lag_min": 0, "lag_max": 0},
        "model": {"window_size": 5},
    }
    align(s, cfg)
    window(s, cfg)
    records = s.provenance["train"]
    assert len(records) == 2
    assert records[0]["target_indices"][-1] == 29
    assert records[1]["target_indices"][0] == 34
    assert sum(r["end"] - r["start"] for r in records) == 62


@pytest.mark.parametrize("kind", ["cob", "five_section"])
def test_kdf_layouts(kind, tmp_path):
    n = 12
    x = np.arange(n * 8).reshape(n, 8)
    if kind == "cob":
        header = [1, 5, 2]
    else:
        header = [1, 3, 2, 1, 1]
    path = tmp_path / "data.kdf"
    with path.open("wb") as f:
        np.asarray(header, "<f4").tofile(f)
        x.astype("<f4").tofile(f)
    features, target, time = read_kdf(path, kind)
    np.testing.assert_array_equal(features, x[:, 1 : 1 + header[1]])
    np.testing.assert_array_equal(target, x[:, 1 + header[1] : 1 + header[1] + 2])
    np.testing.assert_array_equal(time, x[:, 0])


def test_ambiguous_header_requires_choice(tmp_path):
    # Both a 3-field 1+1+1 and five-field 1+1+1+1+1 layout divide this payload.
    path = tmp_path / "ambiguous.kdf"
    np.ones(15, dtype="<f4").tofile(path)
    with pytest.raises(ValueError, match="ambiguous"):
        read_kdf(path, "auto")
    assert read_kdf(path, "cob")[0].shape == (4, 1)
    assert read_kdf(path, "five_section")[0].shape == (2, 1)


def test_constant_clock_rejected():
    s = fixture_state()
    f = s.train_files[0]
    s.train_files[0] = KdfFile(f.path, f.features, f.kinematics, np.ones(70))
    with pytest.raises(ValueError, match="timestamps"):
        align(s, {})

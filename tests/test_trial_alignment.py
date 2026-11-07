"""Independent trial delays, split isolation, fallbacks, and source chronology."""

import json
from copy import deepcopy

import numpy as np
import pytest

from unrbl_ml_pipeline.training.alignment_report import plot_trial_alignment
from unrbl_ml_pipeline.training.mamba_data import validate_metadata
from unrbl_ml_pipeline.training.preprocessing import list_steps, run_pipeline
from unrbl_ml_pipeline.training.preprocessing._types import KdfFile, PipelineState
from unrbl_ml_pipeline.training.preprocessing.align_lag import run as align_global
from unrbl_ml_pipeline.training.preprocessing.align_trials import run as align_trials
from unrbl_ml_pipeline.training.preprocessing.split_by_detected_trial_deadzone import (
    run as split,
)
from unrbl_ml_pipeline.training.preprocessing.trial_detect import (
    extend_trials_with_deadzone,
)
from unrbl_ml_pipeline.training.preprocessing.window import run as window


def trial(delay=0, name="trial_1", n=110, cue=40, stop=65):
    baseline = 1 + (np.arange(n) % 2) * 0.01
    x = np.column_stack((baseline, baseline * 2))
    x[max(0, cue + delay):min(n, stop + delay)] += [1, 2]
    y = np.zeros((n, 2))
    y[cue:stop, 0] = np.linspace(0.5, 1, stop - cue)
    y[cue:stop, 1] = -y[cue:stop, 0]
    return KdfFile(name, x, y, np.arange(n) * 992, trial_id=name,
                   trial_info={"kind": "trial", "trial_id": name,
                               "cue_start": cue, "cue_end": stop})


def config(lag=0):
    return {"alignment": {"lag": lag, "lag_min": -3, "lag_max": 3},
            "model": {"window_size": 5, "prediction_lag_samples": 1}}


@pytest.mark.parametrize("lag", [-3, 0, 3])
def test_independent_residual_delays_and_provenance(lag):
    shifts = [-5, 0, 6]
    originals = [trial(d - lag, f"trial_{i}") for i, d in enumerate(shifts)]
    state = PipelineState(train_files=deepcopy(originals),
                          val_files=[trial()], test_files=[trial()])
    cfg = config(lag)
    align_global(state, cfg)
    globally_aligned = deepcopy(state.train_files)
    align_trials(state, cfg)
    records = state.fitted_preprocessing["trial_alignment"]["trials"]
    assert [r["residual_shift"] for r in records] == shifts
    assert [r["status"] for r in records] == ["corrected", "unchanged", "corrected"]
    for original, before, after, d in zip(originals, globally_aligned, state.train_files, shifts):
        np.testing.assert_array_equal(after.kinematics, original.kinematics[after.source_indices])
        np.testing.assert_array_equal(after.features, original.features[after.feature_indices])
        a, b = max(0, d), min(len(before.features), len(before.features) + d)
        np.testing.assert_array_equal(after.nip_time, before.nip_time[a:b])
        assert len(after.features) == len(before.features) - abs(d)
        # The complete nonzero trajectory survives, including all DOFs and signs.
        np.testing.assert_array_equal(after.kinematics[np.any(after.kinematics != 0, axis=1)],
                                      original.kinematics[40:65])
    window(state, cfg)
    validate_metadata(state, cfg)
    for r, d in zip(state.provenance["train"], shifts):
        np.testing.assert_array_equal(r["target_indices"] - r["input_end_indices"], lag - d + 1)
        assert r["trial_info"]["residual_shift"] == d
    json.dumps(state.fitted_preprocessing, allow_nan=False)


@pytest.mark.parametrize("case,reason", [
    ("missing", "missing_onset"), ("ambiguous", "ambiguous_onset"),
    ("constant", "no_usable_channels"), ("nonfinite", "no_usable_channels"),
    ("baseline", "insufficient_baseline"), ("bound", "search_bound_reached"),
    ("negative_bound", "search_bound_reached"),
    ("truncated", "shift_truncates_movement"),
    ("short", "insufficient_window_samples"),
])
def test_fallbacks_preserve_global_data(case, reason):
    f, cfg = trial(), config()
    if case == "missing":
        f.features[40:65] -= [1, 2]
    elif case == "ambiguous":
        f.features[40:65] -= [1, 2]
        f.features[36:40] += [1, 2]
        f.features[45:50] += [1, 2]
    elif case == "constant":
        f.features[:] = 1
    elif case == "nonfinite":
        f.features[:] = np.nan
    elif case == "baseline":
        f = trial(cue=12)
    elif case == "bound":
        f = trial(10)
    elif case == "negative_bound":
        f = trial(-10)
    elif case == "truncated":
        f = trial(8, n=68)
    elif case == "short":
        cfg["model"]["window_size"] = 110
    # Explicit fitted global state avoids asking the global scorer to fit NaNs.
    state = PipelineState(train_files=[f], fitted_preprocessing={"alignment": {"lag": 0}})
    before = deepcopy(f)
    align_trials(state, cfg)
    r = state.fitted_preprocessing["trial_alignment"]["trials"][0]
    assert r["reason"] == reason
    assert r["status"] == "fallback" and r["residual_shift"] == 0
    for attr in ("features", "kinematics", "nip_time", "source_indices", "feature_indices"):
        np.testing.assert_array_equal(getattr(state.train_files[0], attr), getattr(before, attr))
    json.dumps(state.fitted_preprocessing, allow_nan=False)


def test_bad_channels_are_excluded_without_losing_good_onset():
    f = trial(4)
    f.features = np.column_stack((f.features, np.ones(110), np.full(110, np.nan)))
    state = PipelineState(train_files=[f], fitted_preprocessing={"alignment": {"lag": 0}})
    align_trials(state, config())
    r = state.fitted_preprocessing["trial_alignment"]["trials"][0]
    assert r["usable_channels"] == 2 and r["residual_shift"] == 4


def test_timestamp_gaps_preserve_metadata_and_prevent_cross_gap_correction():
    f = trial(3)
    f.nip_time[50:] += 10000
    state = PipelineState(train_files=[f])
    align_global(state, config())
    assert len(state.train_files) == 2
    align_trials(state, config())
    assert {r["reason"] for r in state.fitted_preprocessing["trial_alignment"]["trials"]} == {
        "incomplete_movement_context"}
    window(state, config())
    assert len(state.provenance["train"]) == 2
    assert state.provenance["train"][0]["target_indices"][-1] < 50
    assert state.provenance["train"][1]["input_end_indices"][0] >= 50


def recording():
    n = 1300
    x = np.tile((1 + (np.arange(n) % 2) * 0.01)[:, None], (1, 2))
    y = np.zeros((n, 2))
    for i in range(8):
        start = 60 + 150 * i
        y[start:start + 30, 0] = 1
        x[start + i - 4:start + 30 + i - 4] += 1
    return KdfFile("recording.kdf", x, y, np.arange(n) * 992)


def split_config():
    cfg = config()
    cfg["split"] = {"test_size": 0.2, "val_size": 0.2,
                    "trial_detection": {"pre_cue_samples": 40}}
    cfg["feature_selection"] = {"count": 2}
    return cfg


def test_padding_partition_and_split_assignments():
    a, b = PipelineState(files=[recording()]), PipelineState(files=[recording()])
    cfg = split_config()
    split(a, cfg)
    cfg["split"]["trial_detection"]["pre_cue_samples"] = 0
    split(b, cfg)
    seen = set()
    for name in ("train", "val", "test"):
        af, bf = getattr(a, name + "_files"), getattr(b, name + "_files")
        assert [f.trial_id for f in af] == [f.trial_id for f in bf]
        for f in af:
            rows = set(f.source_indices)
            assert not seen & rows
            seen |= rows
            if f.trial_info["kind"] == "trial":
                assert f.trial_info["cue_start"] - f.source_indices[0] == 40
    assert seen == set(range(1300))
    adjacent = [{"start": 5, "end": 15}, {"start": 15, "end": 20},
                {"start": 25, "end": 30}]
    padded = extend_trials_with_deadzone(adjacent, 40, 40)
    assert [(t["start"], t["end"]) for t in padded] == [(0, 15), (15, 23), (23, 40)]


def test_toggle_keeps_heldout_arrays_and_provenance_identical():
    cfg = split_config()
    steps = ["split_by_detected_trial_deadzone", "align_lag", "select_features", "window"]
    a = run_pipeline(steps, cfg, PipelineState(files=[recording()]))
    b = run_pipeline(steps[:2] + ["align_trials"] + steps[2:], cfg,
                     PipelineState(files=[recording()]))
    assert a.fitted_preprocessing["alignment"] == b.fitted_preprocessing["alignment"]
    for split_name in ("val", "test"):
        for prefix in ("X_", "Y_", "trial_id_", "source_file_"):
            np.testing.assert_array_equal(getattr(a, prefix + split_name), getattr(b, prefix + split_name))
        for ar, br in zip(a.provenance[split_name], b.provenance[split_name]):
            assert ar.keys() == br.keys()
            for key in ar:
                if isinstance(ar[key], np.ndarray):
                    np.testing.assert_array_equal(ar[key], br[key])
                else:
                    assert ar[key] == br[key]


def test_heldout_perturbations_do_not_change_training_fit():
    a = PipelineState(train_files=[trial(2)], val_files=[trial()], test_files=[trial()])
    b = deepcopy(a)
    b.val_files[0].features *= 100
    b.test_files[0].kinematics *= -100
    cfg = config()
    cfg["alignment"].pop("lag")
    for state in (a, b):
        align_global(state, cfg)
        align_trials(state, cfg)
    assert a.fitted_preprocessing == b.fitted_preprocessing
    np.testing.assert_array_equal(a.train_files[0].kinematics, b.train_files[0].kinematics)


def test_ordering_metadata_and_settings_errors():
    assert "align_trials" in list_steps()
    with pytest.raises(ValueError, match="Invalid preprocessing step"):
        run_pipeline(["split_by_detected_trial_deadzone", "align_lag", "align_trials", None], {})
    for steps in (["align_trials"],
                  ["split_by_detected_trial_deadzone", "align_trials", "align_lag"],
                  ["split_by_detected_trial_deadzone", "align_lag", "normalize", "align_trials"]):
        with pytest.raises(ValueError, match="align_trials"):
            run_pipeline(steps, {})
    with pytest.raises(ValueError, match="align_lag"):
        align_trials(PipelineState(), {})
    f = trial()
    f.trial_info = {}
    with pytest.raises(ValueError, match="cue metadata"):
        align_trials(PipelineState(train_files=[f], fitted_preprocessing={"alignment": {"lag": 0}}), {})
    for key, value in (("baseline_samples", 0), ("min_active_samples", 1.5),
                       ("max_residual_lag_samples", True), ("threshold_multiplier", float("nan"))):
        with pytest.raises(ValueError, match=key):
            align_trials(PipelineState(), {"trial_alignment": {key: value}})


def test_bounded_diagnostics_and_plot(tmp_path):
    state = PipelineState(train_files=[trial(4, f"trial_{i}") for i in range(20)] +
                          [trial(cue=12, name="fallback")])
    align_global(state, config())
    align_trials(state, config())
    assert len(state.alignment_examples) == 12
    assert {e["record"]["status"] for e in state.alignment_examples} == {"corrected", "fallback"}
    path = tmp_path / "trial_alignment.png"
    assert plot_trial_alignment(path, state)
    assert path.stat().st_size > 1000


def test_recorded_run_saves_alignment_artifacts_and_checks_resume(tmp_path):
    pytest.importorskip("tensorflow")
    from unrbl_ml_pipeline.training import run_training
    from unrbl_ml_pipeline.training.mamba_train import fingerprint

    f = recording()
    data = tmp_path / "data"
    data.mkdir()
    with (data / "recording.kdf").open("wb") as stream:
        np.asarray([1, 2, 2], dtype="<f4").tofile(stream)
        np.column_stack((f.nip_time, f.features, f.kinematics)).astype("<f4").tofile(stream)
    cfg = split_config()
    cfg.update(model_name="mamba_decoder", epochs=1, min_epochs=1, state_mode="reset",
               batch_size=128, loss={"type": "mse"},
               architecture={"mamba_width": 4, "state_size": 2, "head_units": 4},
               data={"base_dir": str(data), "kdf_format": "cob", "beta_threshold": 0},
               preprocessing_steps=["load_kdf", "quality_filter", "split_by_detected_trial_deadzone",
                                    "align_lag", "align_trials", "select_features", "window"])
    _, result = run_training(config=cfg, output_dir=tmp_path / "runs", verbose=0)
    root = result["run_dir"]
    records = json.loads((root / "preprocessing.json").read_text())["trial_alignment"]
    assert records["counts"]["corrected"] > 0
    assert (root / "trial_alignment.png").stat().st_size > 1000
    assert "trial_alignment.png" in (root / "performance.md").read_text()
    assert json.loads((root / "config.json").read_text())["model_name"] == "mamba_decoder"
    changed = deepcopy(result["config"])
    changed["preprocessing_steps"].remove("align_trials")
    assert fingerprint(result["state"], changed) != fingerprint(result["state"], result["config"])

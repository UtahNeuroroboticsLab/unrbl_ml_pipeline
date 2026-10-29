"""KDF selection, configuration, direct training, and recorded-run integration."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from unrbl_ml_pipeline import models
from unrbl_ml_pipeline.training import resolve_config, run_training, train
from unrbl_ml_pipeline.training.config import deep_merge, load_jsonc
from unrbl_ml_pipeline.training.preprocessing import run_pipeline
from unrbl_ml_pipeline.training.preprocessing._types import PipelineState
from unrbl_ml_pipeline.training.preprocessing.load_kdf import recording_date
from unrbl_ml_pipeline.training.preprocessing.quality_filter import (
    correlation_quality_score,
)

ROOT = Path(__file__).resolve().parents[1]


def write_kdf(path, features, targets):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.array([1, features.shape[1], targets.shape[1]], dtype="<f4")
    rows = np.column_stack((np.arange(len(features)), features, targets)).astype("<f4")
    with path.open("wb") as stream:
        header.tofile(stream)
        rows.tofile(stream)


@pytest.fixture
def recordings(tmp_path):
    rng = np.random.RandomState(10)
    for i in range(8):
        targets = rng.normal(size=(28, 2)).astype("float32")
        features = np.column_stack((targets, rng.normal(size=(28, 2)))).astype("float32")
        write_kdf(tmp_path / "movement" / f"trainKDF_202608{i + 1:02d}-120000.kdf", features, targets)
    return tmp_path


def small_config(recordings):
    return {
        "data": {"base_dir": str(recordings), "beta_threshold": 0.2},
        "preprocessing_steps": ["load_kdf", "quality_filter", "split_by_file", "normalize", "window"],
        "model": {"window_size": 3},
        "architecture": {"conv_filters": 3, "conv_kernel_size": 2, "lstm_units": 4,
                         "dense_units": 4, "use_batch_norm": False},
        "epochs": 1, "batch_size": 16,
    }


def test_jsonc_strings_aliases_and_precedence(tmp_path):
    path = tmp_path / "config.jsonc"
    path.write_text('''{
      // Comments and trailing commas are allowed.
      "data": {"base_dir": "recordings", "beta_threshold": 0.4},
      "url": "https://example.invalid/a/*b*/",
      "literal": ",}",
      "window_size": 7,
      "output_dir": "runs",
      "model_overrides": {"clstm_decoder": {"architecture": {"conv_filters": 12}}},
    }''')
    default_before = deepcopy(models.clstm_decoder.default_config)
    overrides = {"beta_threshold": 0.3, "window_size": 5}
    original = deepcopy(overrides)
    cfg = resolve_config("clstm_decoder", config_path=path, user_config=overrides)
    assert cfg["model"]["window_size"] == 5
    assert cfg["data"]["beta_threshold"] == 0.3
    assert cfg["architecture"]["conv_filters"] == 12
    assert cfg["data"]["base_dir"] == str(tmp_path / "recordings")
    assert cfg["output_dir"] == str(tmp_path / "runs")
    assert cfg["url"] == "https://example.invalid/a/*b*/"
    assert cfg["literal"] == ",}"
    cfg["preprocessing_steps"].append("not_real")
    assert overrides == original
    assert models.clstm_decoder.default_config == default_before
    assert load_jsonc(path)["window_size"] == 7


def test_model_switch_preserves_model_defaults():
    path = ROOT / "config_train.jsonc"
    clstm = resolve_config("clstm_decoder", config_path=path)
    caleb = resolve_config("caleb_decoder", config_path=path)
    kf = resolve_config("kf_decoder", config_path=path)
    assert clstm["model"]["window_size"] == 10
    assert clstm["architecture"]["conv_filters"] == 64
    assert clstm["architecture"]["lstm_units"] == 128
    assert clstm["stateful"] is False
    assert not clstm["architecture"]["use_deadband_activation_output"]
    assert not clstm["postprocessing"]["deadband"]["enabled"]
    assert caleb["model"]["window_size"] == 1
    assert caleb["optimizer"] == "sgd"
    assert "conv_filters" not in caleb["architecture"]
    assert kf["preprocessing_steps"] == ["load_kdf", "quality_filter", "split_by_file"]


def test_dates_selection_and_load_errors(recordings):
    original = next(recordings.rglob("*.kdf"))
    (recordings / "prefix_trainKDF_20260802-120000.kdf").write_bytes(original.read_bytes())
    (recordings / "undated.kdf").write_bytes(original.read_bytes())
    (recordings / "trainKDF_20260803-120001.kdf").write_bytes(b"broken")
    (recordings / "trainKDF_20260802-120000.kdf:Zone.Identifier").write_text("ignored")
    cfg = {"data": {"base_dir": str(recordings), "date_from": "2026-08-02",
                    "date_to": "2026-08-03", "beta_threshold": 0.2}}
    state = run_pipeline(["load_kdf", "quality_filter"], cfg)
    assert len(state.files) == 3
    assert len(state.selection_report) == 11
    assert sum(r["status"] == "load_error" for r in state.selection_report) == 1
    assert all(r["score"] >= 0.2 for r in state.selection_report if r["status"] == "kept")
    assert recording_date("prefix_trainKDF_20260802-120000.kdf").isoformat() == "2026-08-02"
    assert recording_date("trainKDF_20260230-120000.kdf") is None
    assert recording_date("trainKDF_20260802-990000.kdf") is None
    assert len(run_pipeline(["load_kdf"], {"data": {"base_dir": str(recordings)}}).files) == 10


@pytest.mark.parametrize("bounds", [
    {"date_from": "2026-8-01"}, {"date_to": "2026-02-30"},
    {"date_from": "2026-09-01", "date_to": "2026-08-01"},
    {"date_from": "2030-01-01"},
])
def test_invalid_or_empty_date_selection(recordings, bounds):
    with pytest.raises(ValueError):
        run_pipeline(["load_kdf"], {"data": dict(bounds, base_dir=str(recordings))})


def test_quality_threshold_equality_and_rejection(recordings):
    cfg = {"data": {"base_dir": str(recordings)}}
    state = run_pipeline(["load_kdf"], cfg)
    f = state.files[0]
    beta, _ = correlation_quality_score(f.features, f.kinematics)
    state.files = [f]
    result = run_pipeline(["quality_filter"], {"data": {"beta_threshold": beta}}, state)
    assert result.files == [f]
    f.features[:] = 0
    with pytest.raises(RuntimeError, match="Every candidate"):
        run_pipeline(["quality_filter"], {"data": {"beta_threshold": 0.2}}, state)
    assert any(r["status"] == "rejected" for r in state.selection_report)


def test_dimension_mismatch(recordings):
    write_kdf(recordings / "bad.kdf", np.ones((10, 5)), np.ones((10, 2)))
    with pytest.raises(ValueError, match="Incompatible"):
        run_pipeline(["load_kdf"], {"data": {"base_dir": str(recordings)}})


def test_normalization_windows_and_split_records(recordings):
    cfg = resolve_config("clstm_decoder", user_config=small_config(recordings))
    state = run_pipeline(cfg["preprocessing_steps"], cfg)
    expected = np.concatenate([f.features for f in state.train_files]).mean(axis=0)
    np.testing.assert_allclose(state.norm_stats.feat_mean, expected)
    assert state.X_train.shape[1:] == (3, 4)
    first = state.train_files[0]
    np.testing.assert_array_equal(state.X_train[0], first.features[:3])
    assert len(state.X_train) == sum(len(f.features) - 2 for f in state.train_files)
    assert all(len(r["splits"]) == 1 for r in state.selection_report)
    assert set(state.trial_id_train).isdisjoint(state.trial_id_test)
    assert state.split_info["segments"]["train"]


@pytest.mark.parametrize("override, message", [
    ({"stateful": True}, "Stateful"),
    ({"loss": {"type": "recency_weighted_mse"}}, "unsupported"),
    ({"preprocessing_steps": ["stateful_chunk"]}, "chunk"),
    ({"split": {"val_size": 0.9, "test_size": 0.2}}, "sum"),
])
def test_unsupported_configs_fail_before_loading(override, message):
    with pytest.raises(ValueError, match=message):
        train("clstm_decoder", config=override)


def test_empty_validation_split_is_actionable(monkeypatch, recordings):
    from unrbl_ml_pipeline.training import preprocessing
    monkeypatch.setattr(preprocessing, "run_pipeline", lambda *a: PipelineState(X_train=np.ones((2, 3, 4))))
    with pytest.raises(ValueError, match="empty val"):
        train("clstm_decoder", config=small_config(recordings))


def test_clstm_run_reload_and_artifacts(recordings, tmp_path):
    import keras
    cfg = small_config(recordings)
    cfg["epochs"] = 3
    fitted, result = run_training("clstm_decoder", config=cfg, epochs=1,
                                  output_dir=tmp_path / "outputs", verbose=0)
    assert len(result["history"].history["loss"]) == 1
    assert result["config"]["epochs"] == 1
    assert result["config"]["output_dir"] == str(tmp_path / "outputs")
    assert cfg["epochs"] == 3
    run_dir = result["run_dir"]
    assert {"model.keras", "config.json", "selection.json", "splits.json", "history.json",
            "metrics.json", "run.json", "loss.png", "predictions.png"} <= {p.name for p in run_dir.iterdir()}
    restored = keras.saving.load_model(result["model_path"])
    x = result["state"].X_test[:4]
    np.testing.assert_allclose(fitted(x).numpy(), restored(x).numpy(), atol=1e-6)
    norm = restored.get_layer("feature_normalization")
    expected = (x - result["state"].norm_stats.feat_mean) / result["state"].norm_stats.feat_std
    np.testing.assert_allclose(norm(x).numpy(), expected, atol=1e-5)
    assert not any(isinstance(layer, keras.layers.Lambda) for layer in fitted.layers)
    assert fitted.get_layer("lstm").stateful is False
    assert json.loads((run_dir / "config.json").read_text())["epochs"] == 1
    error = result["Y_pred"] - result["Y_test"]
    assert result["metrics"]["aggregate"]["mse"] == pytest.approx(np.mean(error ** 2))
    assert (run_dir / "predictions.png").stat().st_size > 1000


def test_caleb_direct_train_and_reload(recordings, tmp_path):
    import keras
    cfg = small_config(recordings)
    cfg["model"]["window_size"] = 1
    path = tmp_path / "caleb.keras"
    fitted, results = train("caleb_decoder", config=cfg, save_path=path, epochs=1,
                            batch_size=8, optimizer="adam", learning_rate=0.0002, verbose=0)
    assert fitted.input_shape == (None, 1, 4)
    assert results["config"]["batch_size"] == 8
    assert float(fitted.optimizer.learning_rate.numpy()) == pytest.approx(0.0002)
    loaded = keras.saving.load_model(path)
    x = results["state"].X_test[:2]
    np.testing.assert_allclose(fitted(x).numpy(), loaded(x).numpy(), atol=1e-6)


def test_kalman_run_without_loss_plot(recordings, tmp_path):
    cfg = {"data": {"base_dir": str(recordings)},
           "postprocessing": {"deadband": {"enabled": False}}}
    fitted, results = run_training("kf_decoder", config=cfg, output_dir=tmp_path / "outputs")
    assert "history" not in results
    assert (results["run_dir"] / "model.pkl").is_file()
    assert not (results["run_dir"] / "loss.png").exists()
    from unrbl_ml_pipeline.training.kalman import KalmanFilterDecoder
    loaded = KalmanFilterDecoder.load(results["model_path"])
    features = results["state"].test_files[0].features
    np.testing.assert_allclose(fitted.predict(features), loaded.predict(features))


@pytest.mark.parametrize("filename, wrapped", [("testfile.py", True), ("testfile_train_api.py", False)])
def test_examples_import_safely_and_route(monkeypatch, tmp_path, filename, wrapped):
    from unrbl_ml_pipeline import training
    calls = []

    def fake_train(*args, **kwargs):
        calls.append((args, kwargs))
        return object(), {"history": SimpleNamespace(history={"loss": [1.0]}),
                          "test_loss": 1.0, "model_path": tmp_path / "model.keras",
                          "state": SimpleNamespace(files=[]), "Y_pred": np.ones((2, 6))}

    def wrong_route(*args, **kwargs):
        raise AssertionError("Wrong training API called")

    monkeypatch.setattr(training, "run_training", fake_train if wrapped else wrong_route)
    monkeypatch.setattr(training, "train", wrong_route if wrapped else fake_train)
    spec = importlib.util.spec_from_file_location("training_example", ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not calls
    monkeypatch.chdir(tmp_path)
    module.OVERRIDES = {"epochs": 3}
    assert module.main(["--epochs", "1", "--model", "caleb_decoder", "--output-dir", str(tmp_path)]) == 0
    args, kwargs = calls[0]
    assert (args[0] if wrapped else args[0].name) == "caleb_decoder"
    assert kwargs["config"]["epochs"] == 1
    assert Path(kwargs["config_path"]) == ROOT / "config_train.jsonc"


def test_cli_flags_win_over_python_model_overrides():
    from unrbl_ml_pipeline.cli import parse_training_args, training_overrides
    args = parse_training_args(["--epochs", "1", "--beta", "0.4"])
    cfg = training_overrides(args, {"model_overrides": {"clstm_decoder": {"epochs": 9}}})
    assert cfg["epochs"] == 1
    assert cfg["data"]["beta_threshold"] == 0.4


def test_merge_does_not_share_mutable_values():
    base = {"steps": ["a"]}
    result = deep_merge(base, {})
    result["steps"].append("b")
    assert base == {"steps": ["a"]}


def test_kdf_training_streams_every_phase_across_epochs(monkeypatch, recordings):
    import keras
    import tensorflow as tf

    observed = {}
    for method in ("fit", "predict", "evaluate"):
        original = getattr(keras.Model, method)

        def check_dataset(self, x, *args, _method=method, _original=original, **kwargs):
            assert isinstance(x, tf.data.Dataset)
            assert kwargs.get("batch_size") is None
            assert kwargs.get("y") is None
            observed[_method] = True
            if _method == "fit":
                assert isinstance(kwargs["validation_data"], tf.data.Dataset)
            return _original(self, x, *args, **kwargs)

        monkeypatch.setattr(keras.Model, method, check_dataset)

    class EpochBatches(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.counts = []

        def on_epoch_begin(self, epoch, logs=None):
            self.counts.append(0)

        def on_train_batch_end(self, batch, logs=None):
            self.counts[-1] += 1

    callback = EpochBatches()
    cfg = small_config(recordings)
    _, result = train("clstm_decoder", config=cfg, epochs=2, verbose=0, callbacks=[callback])
    n = len(result["state"].X_train)
    expected_batches = (n + cfg["batch_size"] - 1) // cfg["batch_size"]
    assert callback.counts == [expected_batches, expected_batches]
    assert set(observed) == {"fit", "predict", "evaluate"}
    assert len(result["history"].history["val_loss"]) == 2
    assert np.isfinite(result["history"].history["val_loss"]).all()
    assert result["test_loss"] == pytest.approx(
        np.mean((result["Y_pred_raw"] - result["Y_test"]) ** 2), rel=1e-5,
    )

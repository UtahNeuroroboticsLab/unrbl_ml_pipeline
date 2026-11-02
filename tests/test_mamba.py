"""Mamba public contracts, numerical state behavior and training regressions."""

import json
import subprocess
import sys

import numpy as np
import pytest

from unrbl_ml_pipeline import models

KINDS = ("mamba", "tcn_mamba", "mamba_tcn", "mamba_tcl", "mamba_lstm")
ARCH = {
    "mamba_width": 8,
    "state_size": 4,
    "tcn_width": 8,
    "lstm_units": 8,
    "head_units": 8,
    "dropout": 0.2,
}


@pytest.fixture
def kdfs(tmp_path):
    rng = np.random.default_rng(12)
    for i in range(8):
        x = rng.normal(size=(36, 5)).astype("float32")
        y = np.tanh(x[:, :2])
        payload = np.column_stack((np.arange(36) * 1000, x, y)).astype("<f4")
        with (tmp_path / f"trainKDF_202610{i + 10}-000000.kdf").open("wb") as f:
            np.asarray([1, 5, 2], dtype="<f4").tofile(f)
            payload.tofile(f)
    return tmp_path


def config():
    return {
        "preprocessing_steps": [
            "load_kdf",
            "quality_filter",
            "split_by_file",
            "align_lag",
            "select_features",
            "window",
        ],
        "data": {
            "beta_threshold": 0,
            "kinematic_column_indices": [0, 1],
            "dof_names": ["a", "b"],
            "kdf_format": "cob",
        },
        "model": {"window_size": 5, "prediction_lag_samples": 1},
        "architecture": ARCH,
        "alignment": {"lag": 0, "lag_min": 0, "lag_max": 0},
        "feature_selection": {"count": 4},
        "epochs": 1,
        "batch_size": 8,
        "streams": 2,
        "tbptt_steps": 8,
    }


def test_lightweight_discovery():
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            (
                "import sys; from unrbl_ml_pipeline.training import supported_losses; "
                "assert len(supported_losses('mamba_decoder'))==4; "
                "assert 'tensorflow' not in sys.modules and 'keras' not in sys.modules"
            ),
        ]
    )


@pytest.mark.parametrize("kind", KINDS)
def test_state_and_reload(kind, tmp_path):
    import keras
    import tensorflow as tf

    from unrbl_ml_pipeline.inference import load_model_file

    keras.utils.set_random_seed(21)
    m = models.get(kind + "_decoder").build((5, 5), 2, architecture=ARCH)
    summary = []
    m.summary(print_fn=summary.append)
    assert "?" not in "\n".join(summary)
    block_output = m.mamba(tf.zeros((2, 3, m.width)))
    declared = m.mamba.compute_output_shape((2, 3, m.width))
    actual = tf.nest.map_structure(lambda t: tuple(t.shape), block_output)
    assert actual == declared
    assert m.mamba.compute_output_shape((None, None, m.width))[0] == (
        None,
        None,
        m.width,
    )
    for norm in (m.feature_norm, m.fusion_norm):
        width = int(norm.gamma.shape[0])
        assert norm.compute_output_shape((None, width)) == (None, width)
        assert tuple(norm(tf.zeros((2, width))).shape) == (2, width)
    x = np.random.default_rng(2).normal(size=(2, 14, 5)).astype("float32")
    state = m.prefill(x[:, :4])
    whole, end = m.step(x[:, 4:], state)
    first, state = m.step(x[:, 4:8], state)
    second, state = m.step(x[:, 8:], state)
    np.testing.assert_allclose(whole, np.concatenate((first, second), 1), atol=2e-6)
    for a, b in zip(end, state):
        np.testing.assert_allclose(a, b, atol=2e-6)
    np.testing.assert_allclose(whole[:, 0], m.predict_windows(x[:, :5]), atol=2e-6)
    m.save(tmp_path / "model.keras")
    loaded = load_model_file(tmp_path / "model.keras")
    np.testing.assert_allclose(
        loaded.step(x[:, 4:], loaded.prefill(x[:, :4]))[0], whole, atol=2e-6
    )
    stream = loaded.stream()
    actual = np.array([stream.predict(x[0, i - 4 : i + 1]) for i in range(4, 14)])
    np.testing.assert_allclose(actual, whole[0], atol=2e-6)
    with pytest.raises(ValueError, match="explicit state"):
        m(x[:, :5])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", ["carry", "reset"])
def test_small_training(kind, mode, kdfs, tmp_path):
    from unrbl_ml_pipeline.training import train

    cfg = config()
    cfg["state_mode"] = mode
    _m, r = train(
        kind + "_decoder",
        data_dir=kdfs,
        config=cfg,
        save_path=tmp_path / "m.keras",
        verbose=0,
    )
    assert len(r["history"].history["loss"]) == 1
    assert np.isfinite(r["test_loss"])
    assert r["Y_pred"].shape == r["Y_test"].shape
    assert (tmp_path / "checkpoints/latest.json").is_file()


def test_regularized_resume_and_reports(kdfs, tmp_path):
    from unrbl_ml_pipeline.training import run_training

    cfg = config()
    cfg["loss"] = {"type": "mse_pinn_jacobian", "warmup_epochs": 0}
    cfg["epochs"] = 1
    _, r = run_training(
        "mamba_lstm_decoder",
        data_dir=kdfs,
        config=cfg,
        output_dir=tmp_path / "runs",
        verbose=0,
    )
    assert r["calibration"]["train_only"]
    assert r["history"].history["jacobian"][0] > 0
    cfg.update(epochs=2, resume_from=str(r["checkpoint_dir"]))
    _resumed, rr = run_training(
        "mamba_lstm_decoder",
        data_dir=kdfs,
        config=cfg,
        output_dir=tmp_path / "runs",
        verbose=0,
    )
    del cfg["resume_from"]
    _full, fr = run_training(
        "mamba_lstm_decoder",
        data_dir=kdfs,
        config=cfg,
        output_dir=tmp_path / "runs",
        verbose=0,
    )
    np.testing.assert_allclose(rr["Y_pred"], fr["Y_pred"], atol=1e-6)
    for key in rr["history"].history:
        if key == "epoch_seconds":
            assert len(rr["history"].history[key]) == 2
            assert np.all(np.asarray(rr["history"].history[key]) > 0)
            continue  # Wall-clock durations are intentionally not deterministic.
        np.testing.assert_allclose(
            rr["history"].history[key], fr["history"].history[key], atol=1e-6
        )
    assert (rr["run_dir"] / "performance.md").is_file()
    assert json.loads((rr["run_dir"] / "calibration.json").read_text())["train_only"]


@pytest.mark.parametrize("mode,verbose", [("carry", 1), ("reset", 2)])
def test_epoch_display_metrics(mode, verbose, kdfs, capsys):
    import keras

    from unrbl_ml_pipeline.training import train
    from unrbl_ml_pipeline.training.mamba_data import predict

    class Capture(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            self.epoch_logs = dict(logs)

    capture = Capture()
    cfg = config()
    cfg["state_mode"] = mode
    model, result = train(
        "mamba_decoder",
        data_dir=kdfs,
        config=cfg,
        verbose=verbose,
        callbacks=[capture],
    )
    output = capsys.readouterr().out
    assert "Epoch 1/1" in output
    assert "sample" in output
    assert "?" not in output
    history = result["history"].history
    for name in (
        "loss",
        "mse",
        "rmse",
        "objective",
        "val_loss",
        "val_mse",
        "val_rmse",
        "learning_rate",
        "epoch_seconds",
    ):
        assert name + ":" in output
        assert history[name][0] == capture.epoch_logs[name]
        assert np.isfinite(history[name][0])
    assert history["mse"] == history["loss"]
    assert history["val_mse"] == history["val_loss"]
    np.testing.assert_allclose(history["rmse"], np.sqrt(history["mse"]))
    prediction = predict(model, result["state"], "val", result["config"])
    expected = np.mean((prediction.astype(np.float64) - result["state"].Y_val) ** 2)
    np.testing.assert_allclose(history["val_mse"][0], expected)
    np.testing.assert_allclose(history["val_rmse"][0], np.sqrt(expected))
    assert history["epoch_seconds"][0] > 0


def test_compatibility_rejection(kdfs):
    from unrbl_ml_pipeline.training import supported_losses, train

    assert supported_losses("kf_decoder") == []
    for name, loss in [
        ("mamba_decoder", "huber"),
        ("clstm_decoder", "mse_pinn"),
        ("kf_decoder", "mse_jacobian"),
    ]:
        with pytest.raises(ValueError, match="unsupported"):
            train(
                name,
                data_dir=kdfs / "missing",
                config={"loss": {"type": loss}},
                verbose=0,
            )

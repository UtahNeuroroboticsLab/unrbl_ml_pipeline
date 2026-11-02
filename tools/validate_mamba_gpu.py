"""Bounded GPU validation: second derivatives and regularized epoch resume."""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

physical = tf.config.list_physical_devices("GPU")
if not physical:
    raise SystemExit("UNVERIFIED: no accessible TensorFlow GPU")
tf.config.set_logical_device_configuration(
    physical[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
)
from unrbl_ml_pipeline import models
from unrbl_ml_pipeline.training import train

start = time.perf_counter()
report = {"device": str(physical[0]), "derivatives": {}, "resume": {}}
a = {
    "mamba_width": 8,
    "tcn_width": 8,
    "state_size": 4,
    "lstm_units": 8,
    "head_units": 8,
    "dropout": 0.2,
}
for kind in ("mamba", "tcn_mamba", "mamba_tcn", "mamba_tcl", "mamba_lstm"):
    model = models.get(kind + "_decoder").build((5, 5), 2, architecture=a)
    state = model.prefill(tf.ones((2, 4, 5)))
    z = tf.random.normal((2, 1, 5))
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            inner.watch(z)
            prediction, _ = model.step(z, state)
            value = tf.reduce_sum(prediction)
        g = inner.gradient(value, z)
        penalty = tf.reduce_sum(g * g)
    derivatives = outer.gradient(penalty, model.trainable_variables)
    assert all(np.isfinite(v.numpy()).all() for v in derivatives if v is not None)
    report["derivatives"][kind] = float(penalty)
with tempfile.TemporaryDirectory(prefix="mamba-gpu-") as directory:
    root = Path(directory)
    data = root / "data"
    data.mkdir()
    rng = np.random.default_rng(4)
    for i in range(8):
        x = rng.normal(size=(20, 5)).astype("float32")
        y = np.tanh(x[:, :2])
        with (data / f"trainKDF_202610{i + 10}-000000.kdf").open("wb") as f:
            np.asarray([1, 5, 2], "<f4").tofile(f)
            np.column_stack((np.arange(20) * 1000, x, y)).astype("<f4").tofile(f)
    for mode in ("carry", "reset"):
        cfg = {
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
                "kdf_format": "cob",
            },
            "alignment": {"lag": 0, "lag_min": 0, "lag_max": 0},
            "model": {"window_size": 5, "prediction_lag_samples": 1},
            "architecture": a,
            "streams": 2,
            "tbptt_steps": 8,
            "batch_size": 16,
            "state_mode": mode,
            "loss": {"type": "mse_pinn_jacobian", "warmup_epochs": 0},
            "epochs": 1,
        }
        _, first = train(
            "mamba_decoder",
            data_dir=data,
            config=cfg,
            save_path=root / mode / "model.keras",
            verbose=0,
        )
        cfg.update(epochs=2, resume_from=str(first["checkpoint_dir"]))
        _, resumed = train(
            "mamba_decoder",
            data_dir=data,
            config=cfg,
            save_path=root / (mode + "_resume") / "model.keras",
            verbose=0,
        )
        del cfg["resume_from"]
        _, full = train(
            "mamba_decoder",
            data_dir=data,
            config=cfg,
            save_path=root / (mode + "_full") / "model.keras",
            verbose=0,
        )
        delta = float(np.max(np.abs(resumed["Y_pred"] - full["Y_pred"])))
        np.testing.assert_allclose(
            resumed["Y_pred"], full["Y_pred"], atol=2e-5, rtol=1e-4
        )
        report["resume"][mode] = {
            "max_abs_difference": delta,
            "calibration_train_only": resumed["calibration"]["train_only"],
        }
report["seconds"] = time.perf_counter() - start
report["gpu_memory"] = tf.config.experimental.get_memory_info("GPU:0")
Path("outputs/mamba_validation/gpu.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

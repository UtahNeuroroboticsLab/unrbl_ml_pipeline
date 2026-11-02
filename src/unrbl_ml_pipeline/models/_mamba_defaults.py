"""Backend-free Mamba model defaults."""

from copy import deepcopy

OBJECTIVES = ("mse", "mse_pinn", "mse_jacobian", "mse_pinn_jacobian")
DEFAULTS = {
    "preprocessing_steps": [
        "load_kdf",
        "quality_filter",
        "split_by_detected_trial_deadzone",
        "align_lag",
        "select_features",
        "window",
    ],
    "state_mode": "carry",
    "model": {"window_size": 10, "prediction_lag_samples": 1},
    "data": {
        "beta_threshold": 0.2,
        "kdf_format": "auto",
        "require_usable_timestamps": True,
    },
    "alignment": {"lag_min": -20, "lag_max": 20},
    "feature_selection": {"count": 48, "corr_threshold": 0.3, "candidates": "emg"},
    "architecture": {
        "mamba_width": 64,
        "state_size": 16,
        "conv_kernel": 4,
        "expand": 2,
        "tcn_width": 96,
        "lstm_units": 96,
        "head_units": 192,
        "dropout": 0.2,
    },
    "optimizer": "adamw",
    "learning_rate": 0.0005,
    "weight_decay": 0.0001,
    "gradient_clip": 1.0,
    "epochs": 500,
    "min_epochs": 145,
    "early_stopping_patience": 45,
    "batch_size": 64,
    "streams": 4,
    "tbptt_steps": 16,
    "seed": 0,
    "loss": {
        "type": "mse",
        "warmup_epochs": 20,
        "temporal_ratio": 0.1,
        "jacobian_ratio": 0.03,
    },
    "split": {
        "test_size": 0.15,
        "val_size": 0.15,
        "random_state": 42,
        "trial_detection": {
            "min_trial_len": 10,
            "min_peak": 0.5,
            "merge_gap": 0,
            "active_eps": 1e-9,
            "non_trial_policy": "train",
        },
    },
    "postprocessing": {
        "deadband": {"enabled": False},
        "latching_filter": {"enabled": False},
    },
}


def defaults():
    return deepcopy(DEFAULTS)


def build_decoder(kind, input_shape, num_classes, norm_stats=None, **config):
    from unrbl_ml_pipeline.training.mamba_layers import MambaDecoder

    arch = dict(DEFAULTS["architecture"], **config.get("architecture", {}))
    stats = (
        None
        if norm_stats is None
        else {
            key: getattr(norm_stats, key).tolist()
            for key in ("feat_mean", "feat_std", "kin_mean", "kin_std")
        }
    )
    model = MambaDecoder(
        kind=kind,
        raw_features=input_shape[-1],
        num_outputs=num_classes,
        window_size=input_shape[0],
        architecture=arch,
        state_mode=config.get("state_mode", "carry"),
        selected_indices=config.get("selected_indices"),
        norm_stats=stats,
        seed=config.get("seed", 0),
        name=kind + "_decoder",
    )
    model.build((None,) + tuple(input_shape))
    return model

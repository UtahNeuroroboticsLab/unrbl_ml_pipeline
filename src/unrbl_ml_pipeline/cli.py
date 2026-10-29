"""Shared argument handling for unrbl-train and the two training examples."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_training_args(argv=None, *, default_config=None, default_model="clstm_decoder"):
    parser = argparse.ArgumentParser(description="Train an unrbl_ml_pipeline model on KDF recordings.")
    parser.add_argument("--model", "-m", default=default_model, help="Registered model name.")
    parser.add_argument("--config", "-c", default=default_config, help="Package-native JSONC overrides.")
    parser.add_argument("--data-dir", "-d", help="Override the directory searched for KDF files.")
    parser.add_argument("--output-dir", "-o", help="Directory for saved runs.")
    parser.add_argument("--save-path", "-s", help="Explicit model path (additional copy for the reporting runner).")
    parser.add_argument("--kdf-glob", help="Recursive pattern relative to data-dir, e.g. hook*/*.kdf.")
    parser.add_argument("--beta", type=float, help="Minimum correlation quality score.")
    parser.add_argument("--date-from", help="Inclusive recording date, YYYY-MM-DD.")
    parser.add_argument("--date-to", help="Inclusive recording date, YYYY-MM-DD.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--steps", nargs="+", help="Ordered preprocessing step names; replaces model defaults.")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-steps", action="store_true")
    return parser.parse_args(argv)


def show_catalog(args):
    """Print a requested catalog and return whether training should be skipped."""
    if args.list_models:
        from unrbl_ml_pipeline import models
        for name in models.list_models():
            mod = models.get(name)
            print(f"{name}: {mod.description}")
    if args.list_steps:
        from unrbl_ml_pipeline.training.preprocessing import list_steps
        print("\n".join(list_steps()))
    return args.list_models or args.list_steps


def training_overrides(args, python_overrides=None):
    """Merge explicit CLI values over the editable Python settings."""
    from unrbl_ml_pipeline.training.config import deep_merge
    overrides = {key: getattr(args, key) for key in ("epochs", "batch_size", "learning_rate")
                 if getattr(args, key) is not None}
    data = {key: getattr(args, arg) for key, arg in (
        ("kdf_glob", "kdf_glob"), ("beta_threshold", "beta"),
        ("date_from", "date_from"), ("date_to", "date_to"),
    ) if getattr(args, arg) is not None}
    if data:
        overrides["data"] = data
    if args.window_size is not None:
        overrides["model"] = {"window_size": args.window_size}
    if args.steps is not None:
        overrides["preprocessing_steps"] = args.steps
    # Resolve Python per-model sections before adding CLI overrides, so the CLI wins.
    from unrbl_ml_pipeline.training.config import _normalize
    return deep_merge(_normalize(python_overrides or {}, args.model.lower()), overrides)


def save_model_copy(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def main(argv=None):
    args = parse_training_args(argv)
    if show_catalog(args):
        return 0
    from unrbl_ml_pipeline.training import run_training
    overrides = training_overrides(args)
    # Installed CLI has no repository location; defaults are relative to its caller.
    if args.config is None and args.data_dir is None:
        overrides.setdefault("data", {})["base_dir"] = "data/training_data"
    model, _ = run_training(
        args.model, config_path=args.config, config=overrides,
        data_dir=args.data_dir, output_dir=args.output_dir,
    )
    if args.save_path:
        save_model_copy(model, args.save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

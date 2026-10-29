"""CLI entry point for ``unrbl-train``.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Usage::

    unrbl-train --model clstm_decoder --data-dir /data/kdf/
    unrbl-train --model caleb_decoder --config my_overrides.jsonc --data-dir /data/kdf/
    unrbl-train --model kf_decoder --data-dir /data/kdf/ --save-path kf_fitted.pkl
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="unrbl-train",
        description="Train an unrbl_ml_pipeline model.",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Registered model name (e.g. clstm_decoder, kf_decoder).",
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to a .jsonc config file with parameter overrides.",
    )
    parser.add_argument(
        "--data-dir", "-d", default=None,
        help="Base directory containing .kdf data files.",
    )
    parser.add_argument(
        "--save-path", "-s", default=None,
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List registered models and exit.",
    )
    parser.add_argument(
        "--list-steps", action="store_true",
        help="List available preprocessing steps and exit.",
    )

    args = parser.parse_args(argv)

    if args.list_models:
        from unrbl_ml_pipeline import models
        for name in models.list_models():
            mod = models.get(name)
            desc = getattr(mod, "description", "")
            mtype = getattr(mod, "model_type", "keras")
            print(f"  {name:<25s} [{mtype}] {desc}")
        return 0

    if args.list_steps:
        from unrbl_ml_pipeline.training.preprocessing import list_steps
        for step in list_steps():
            print(f"  {step}")
        return 0

    if args.model is None:
        parser.error("--model is required for training.")

    from unrbl_ml_pipeline.training import train

    _model_result, results = train(
        args.model,
        config_path=args.config,
        data_dir=args.data_dir,
        save_path=args.save_path,
    )

    if isinstance(results, dict):
        if "test_loss" in results:
            print("Test loss: {:.6f}".format(results["test_loss"]))
        if "test_mse" in results:
            print("Test MSE:  {:.6f}".format(results["test_mse"]))
    else:
        print("Training complete.")

    if args.save_path:
        print(f"Model saved to: {args.save_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

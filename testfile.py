"""Train on local KDFs and save the model, run records, and diagnostic plots.

    python testfile.py
    python testfile.py --config config_train.jsonc --epochs 1
    python testfile.py --model caleb_decoder --beta 0.2
"""
from pathlib import Path

from unrbl_ml_pipeline.cli import parse_training_args, show_catalog, training_overrides

HERE = Path(__file__).resolve().parent
MODEL = "clstm_decoder"

# Editable Python overrides. These take precedence over config_train.jsonc.
# Put model-specific settings under model_overrides to keep model switching clean.
OVERRIDES = {
    # "epochs": 10,
    # "data": {"beta_threshold": 0.3, "date_from": "2026-08-01"},
    # "model_overrides": {"clstm_decoder": {"architecture": {"conv_filters": 100}}},
}


def main(argv=None):
    args = parse_training_args(argv, default_config=str(HERE / "config_train.jsonc"),
                               default_model=MODEL)
    if show_catalog(args):
        return 0
    from unrbl_ml_pipeline.training import run_training

    model, results = run_training(
        args.model,
        config_path=args.config,
        config=training_overrides(args, OVERRIDES),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    if args.save_path:
        from unrbl_ml_pipeline.cli import save_model_copy
        save_model_copy(model, args.save_path)
    print("Model:", results["model_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Use train() directly; reporting/plots are optional responsibilities of the caller.

    python testfile_train_api.py
    python testfile_train_api.py --config config_train.jsonc --epochs 1
    python testfile_train_api.py --model kf_decoder
"""
from pathlib import Path

from unrbl_ml_pipeline import models
from unrbl_ml_pipeline.cli import parse_training_args, show_catalog, training_overrides

HERE = Path(__file__).resolve().parent
MODEL = "clstm_decoder"

# Editable Python overrides, applied after the JSONC file and before CLI flags.
OVERRIDES = {
    # "epochs": 10,
    # "data": {"kdf_glob": "hook*/*.kdf", "beta_threshold": 0.2},
}
# None keeps each model's default preprocessing. To choose your own, use:
# PREPROCESSING_STEPS = ["load_kdf", "quality_filter", "split_by_file", "normalize", "window"]
PREPROCESSING_STEPS = None


def main(argv=None):
    args = parse_training_args(argv, default_config=str(HERE / "config_train.jsonc"),
                               default_model=MODEL)
    if show_catalog(args):
        return 0
    from unrbl_ml_pipeline.training import resolve_config, train
    from unrbl_ml_pipeline.training.runner import create_run_directory

    python_settings = dict(OVERRIDES)
    if PREPROCESSING_STEPS is not None:
        python_settings["preprocessing_steps"] = PREPROCESSING_STEPS
    
    overrides = training_overrides(args, python_settings)
    selected_model = models.get(args.model)
    resolved = resolve_config(selected_model, user_config=overrides, config_path=args.config)
    
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    else:
        run_dir = create_run_directory(
            args.output_dir or resolved.get("output_dir", HERE / "outputs"),
            "direct_" + selected_model.name,
        )
        suffix = ".pkl" if models.get_model_type(args.model) == "estimator" else ".keras"
        save_path = run_dir / ("model" + suffix)

    # This is the public low-level API: no run_training() wrapper is involved.
    _model, results = train(
        selected_model,
        config_path=args.config,
        config=overrides,
        data_dir=args.data_dir,
        save_path=save_path,
    )
    if "history" in results:
        history = results["history"].history
        print("Final training loss:", history["loss"][-1])
        print("Test loss:", results["test_loss"])
    else:
        print("Test MSE:", results["test_mse"])
    print("Kept recordings:", len(results["state"].files))
    print("Predictions:", results["Y_pred"].shape)
    print("Model saved to:", save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Lightweight declarations for validated training objectives."""

LOSS_REQUIREMENTS = {
    "mse": "regression targets",
    "huber": "regression targets",
    "idle_weighted_mse": "regression targets",
    "mse_pinn": "consecutive samples, timestamps and trial boundaries",
    "mse_jacobian": "input derivatives and second-order parameter gradients",
    "mse_pinn_jacobian": "consecutive samples, timestamps, trial boundaries and second-order gradients",
}
STANDARD_LOSSES = ("mse", "huber", "idle_weighted_mse")
MAMBA_LOSSES = ("mse", "mse_pinn", "mse_jacobian", "mse_pinn_jacobian")


def supported_losses(model, state_mode=None):
    from unrbl_ml_pipeline import models

    mod = models.get(model) if isinstance(model, str) else model
    mode = state_mode or getattr(mod, "default_config", {}).get("state_mode", "reset")
    return list(getattr(mod, "supported_objectives", {}).get(mode, ()))


def validate_loss(model, config):
    mode = config.get("state_mode", "reset")
    loss = config.get("loss", {"type": "mse"})
    name = loss.get("type", "mse") if isinstance(loss, dict) else loss
    if getattr(model, "model_type", "keras") == "estimator" and name == "mse":
        return  # Evaluation metric only; the estimator owns its fitting objective.
    allowed = supported_losses(model, mode)
    if name not in allowed:
        raise ValueError(
            f"{model.name}: loss {name!r} is unsupported in {mode!r} mode. "
            f"Supported: {', '.join(allowed) or 'estimator-defined objective only'}. "
            f"Requirements: {LOSS_REQUIREMENTS.get(name, 'no integrated training route')}"
        )


def compatibility_markdown():
    from unrbl_ml_pipeline import models

    lines = ["| Model | Mode | Objectives |", "|---|---|---|"]
    for name in models.list_models():
        mod = models.get(name)
        for mode, objectives in getattr(mod, "supported_objectives", {}).items():
            lines.append(
                f"| {name} | {mode} | {', '.join(objectives) or 'estimator-defined'} |"
            )
    return "\n".join(lines) + "\n"

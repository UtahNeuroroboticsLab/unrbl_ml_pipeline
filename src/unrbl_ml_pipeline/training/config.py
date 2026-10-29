"""Configuration utilities: JSONC loading and config merging.

Author: Leonardo M. Ferrisi
Date: 2026-10-29
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path


def load_jsonc(path):
    """Load a JSONC file (JSON with ``//`` and ``/* */`` comments)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Match strings first so comments inside quoted paths/URLs survive.
    strings = r'"(?:\\.|[^"\\])*"'
    text = re.sub(
        strings + r"|//[^\n]*|/\*.*?\*/",
        lambda m: m.group() if m.group().startswith('"') else " ",
        text, flags=re.DOTALL,
    )
    text = re.sub(strings + r"|,\s*(?=[}\]])",
                  lambda m: m.group() if m.group().startswith('"') else "", text)
    return json.loads(text)


def deep_merge(base, override):
    """Recursively merge *override* on top of *base* (neither is mutated)."""
    merged = deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = deepcopy(val)
    return merged


def _normalize(config, model_name, relative_to=None):
    """Normalize one override layer before merging with lower priorities."""
    config = deepcopy(config)
    per_model = config.pop("model_overrides", {})
    config = deep_merge(config, per_model.get(model_name, {}))
    for alias, section, key in (
        ("beta_threshold", "data", "beta_threshold"),
        ("window_size", "model", "window_size"),
        ("prediction_lag_samples", "model", "prediction_lag_samples"),
    ):
        if alias in config:
            config.setdefault(section, {}).setdefault(key, config.pop(alias))
    if relative_to is not None:
        data = config.get("data", {})
        if "base_dir" in data:
            data["base_dir"] = str((relative_to / data["base_dir"]).resolve())
        elif "kdf_glob" in data:
            data["base_dir"] = str(relative_to)
        if "output_dir" in config:
            config["output_dir"] = str((relative_to / config["output_dir"]).resolve())
    return config


def resolve_config(model_name_or_module, user_config=None, config_path=None):
    """Build the final config by merging model defaults with user overrides.

    Parameters
    ----------
    model_name_or_module : str or module
        A registered model name or the model module itself.
    user_config : dict, optional
        Direct overrides as a dict.
    config_path : str, optional
        Path to a ``.jsonc`` file whose contents override model defaults.

    Returns
    -------
    dict
        The merged configuration.
    """
    if isinstance(model_name_or_module, str):
        from unrbl_ml_pipeline import models
        mod = models.get(model_name_or_module)
    else:
        mod = model_name_or_module

    name = getattr(mod, "name", "")
    base = _normalize(getattr(mod, "default_config", None) or {}, name)

    if config_path is not None:
        file_cfg = _normalize(load_jsonc(config_path), name,
                              Path(config_path).resolve().parent)
        base = deep_merge(base, file_cfg)

    if user_config is not None:
        base = deep_merge(base, _normalize(user_config, name))

    return base

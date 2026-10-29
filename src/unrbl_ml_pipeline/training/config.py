"""Configuration utilities: JSONC loading and config merging.

Author: Leonardo M. Ferrisi
Date: 2026-10-29
"""
from __future__ import annotations

import json
import re


def load_jsonc(path):
    """Load a JSONC file (JSON with ``//`` and ``/* */`` comments)."""
    with open(path, "r") as f:
        text = f.read()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def deep_merge(base, override):
    """Recursively merge *override* on top of *base* (neither is mutated)."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


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

    base = dict(getattr(mod, "default_config", None) or {})

    if config_path is not None:
        file_cfg = load_jsonc(config_path)
        base = deep_merge(base, file_cfg)

    if user_config is not None:
        base = deep_merge(base, user_config)

    return base

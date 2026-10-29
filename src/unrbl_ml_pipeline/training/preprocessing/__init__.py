"""Composable preprocessing steps for model training pipelines.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Each step is a module in this package exporting a ``run(state, config)``
function.  Models declare an ordered list of step names in their
``default_config["preprocessing_steps"]``, and ``run_pipeline()`` chains
them via a shared ``PipelineState``.
"""
from __future__ import annotations

import importlib
import os
import pkgutil


def list_steps():
    """Return the names of all available preprocessing step modules."""
    import unrbl_ml_pipeline.training.preprocessing as pkg
    candidates = sorted(
        name for _, name, _ in pkgutil.iter_modules(pkg.__path__)
        if not name.startswith("_")
    )
    return [name for name in candidates
            if callable(getattr(importlib.import_module("." + name, __package__), "run", None))]


def run_pipeline(steps, config, state=None):
    """Execute an ordered list of preprocessing step names.

    Parameters
    ----------
    steps : list[str]
        Step module names in execution order.
    config : dict
        Full merged model config (each step reads the keys it needs).
    state : PipelineState, optional
        Initial state; a fresh one is created if not provided.

    Returns
    -------
    PipelineState
        The state after all steps have run.
    """
    from ._types import PipelineState
    if state is None:
        state = PipelineState()
    if not steps:
        raise ValueError("preprocessing_steps must contain an ordered list of step names")
    for step_name in steps:
        if not isinstance(step_name, str) or not step_name.isidentifier() or step_name.startswith("_"):
            raise ValueError(f"Invalid preprocessing step: {step_name!r}")
        mod = importlib.import_module("." + step_name, __package__)
        if not callable(getattr(mod, "run", None)):
            raise TypeError(f"{step_name!r} is not a runnable preprocessing step")
        state = mod.run(state, config)
    for record in state.selection_report:
        record["splits"] = [
            name for name in ("train", "val", "test")
            if any(os.path.abspath(f.path) == record["path"]
                   for f in getattr(state, name + "_files"))
        ]
    state.split_info["segments"] = {
        name: [{"path": os.path.abspath(f.path), "trial_id": f.trial_id,
                "samples": len(f.features)} for f in getattr(state, name + "_files")]
        for name in ("train", "val", "test")
    }
    return state


def __getattr__(name):
    """Lazily import a step module by name."""
    try:
        return importlib.import_module("." + name, __package__)
    except ModuleNotFoundError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

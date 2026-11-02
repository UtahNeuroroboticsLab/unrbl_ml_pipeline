"""Lazy training exports; loss discovery does not load the numerical backend."""
from .compatibility import supported_losses
from .config import load_jsonc, resolve_config

__all__ = ['Estimator', 'load_jsonc', 'resolve_config', 'run_training', 'supported_losses', 'train']


def __getattr__(name):
    import importlib
    modules = {'train': 'train', 'run_training': 'runner', 'Estimator': 'estimator'}
    if name not in modules:
        raise AttributeError(name)
    # Resolve all heavy exports together so submodule imports cannot shadow functions.
    values = {key: getattr(importlib.import_module('.' + module, __name__), key)
              for key, module in modules.items()}
    globals().update(values)
    return values[name]

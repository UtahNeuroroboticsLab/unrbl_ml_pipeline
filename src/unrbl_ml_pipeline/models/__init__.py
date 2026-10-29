"""Model registry.

Author: Leonardo M. Ferrisi
Date: 2026-10-29

Discover and load model modules by name::

    from unrbl_ml_pipeline import models

    models.list_models()          # ['examplemodel']
    models.ExampleModel.build     # the build function
    models.get("examplemodel")    # same module, by string

Each model file exports ``build(input_shape, num_classes)`` returning an
uncompiled ``keras.Model``.  Keras is imported lazily inside ``build()``, so
importing this package never pulls in TensorFlow.
"""

import importlib

# YOU NEED TO MANUALLY UPDATE THESE IF YOU MAKE A NEW MODEL FILE HERE
_REGISTRY = {
    "examplemodel": "examplemodel",
}


def list_models():
    """Return a sorted list of registered model names."""
    return sorted(_REGISTRY)


def get(name):
    """Look up a model module by name (case-insensitive).

    Returns the module object, which has at least a callable ``build``
    attribute and may have ``name``, ``description``, and ``paper``.

    Raises ``KeyError`` if the name is not in the registry.
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(list_models())
        raise KeyError(
            f"{name} is not a registered model. Available: {available}"
        )
    return importlib.import_module("." + _REGISTRY[key], __package__)


def __getattr__(name):
    """Enable ``models.AlexNet`` attribute-style access (PEP 562)."""
    try:
        return get(name)
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

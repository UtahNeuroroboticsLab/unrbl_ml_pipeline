"""Training stack. Requires `pip install "unrbl-ml-pipeline[train]"` (Python >=3.10).

This subpackage is the [train] boundary. The directory IS the dependency line:

  - Modules **here** may import tensorflow/keras at module level. Importing
    `unrbl_ml_pipeline.training` is already a declaration that you installed [train].
  - Modules in the package **root** may not. They must import on a core-only install
    (numpy only) on the Summit's Python 3.7.

Put anything that needs TF, scikit-learn, or matplotlib in here. If you find yourself wanting
one of those from a root module, that module belongs in this subpackage instead.
"""

try:
    import tensorflow  # noqa: F401
except ImportError:
    # Plain ModuleNotFoundError is already clear; this only adds the pointer to the extra.
    raise ImportError(
        "unrbl_ml_pipeline.training needs the training stack: "
        'pip install "unrbl-ml-pipeline[train]" (needs Python >=3.10; TF has no 3.7 or 3.14 wheels)'
    )

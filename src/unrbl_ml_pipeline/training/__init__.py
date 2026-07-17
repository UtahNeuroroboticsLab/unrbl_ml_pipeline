"""Training stack. Requires Python 3.10-3.13 (where TensorFlow has wheels).

This subpackage is the 3.7 boundary. The directory IS the Python-version line:

  - Modules **here** may import tensorflow/keras/sklearn/matplotlib at module level.
    Importing `unrbl_ml_pipeline.training` is already a declaration that you are on 3.10-3.13.
  - Modules in the package **root** may not. They must import on the Summit's Python 3.7,
    where pyproject's markers install numpy + unrbl-tflite-runtime and no TF at all.

Put anything that needs TF, scikit-learn, or matplotlib in here. If you find yourself wanting
one of those from a root module, that module belongs in this subpackage instead.
"""

import sys

try:
    import tensorflow  # noqa: F401
except ImportError:
    # On 3.10-3.13 the markers guarantee TF, so this fires on the Summit's 3.7, in the 3.8/3.9
    # and 3.14+ marker gaps, or on a --no-deps install. Say the unfixable part plainly: on 3.7
    # no pip command puts TF here, and on 3.14 none exists yet.
    raise ImportError(
        "unrbl_ml_pipeline.training needs TensorFlow, which ships wheels only for Python "
        "3.10-3.13 (running {}.{}). Train on an interpreter in that range, convert to "
        ".tflite, and deploy that.".format(sys.version_info[0], sys.version_info[1])
    )

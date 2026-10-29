"""Model loading for deployment targets.

Root module: must import on a 3.7 install (numpy + tflite_runtime, no TF) and stay Python
3.7-syntax-clean.

Backends are imported lazily *inside* each branch on purpose. Which one exists depends on the
interpreter -- pyproject's markers install tensorflow on 3.10-3.13 and the lab's TFLite 2.5.0
build on 3.7 -- and `import unrbl_ml_pipeline` must not drag either one in.

Author: Leonardo M. Ferrisi
Date: 2026-07-16
"""

import os
import sys

TFLITE_SUFFIX = ".tflite"
KERAS_SUFFIX = ".keras"


def load_model_file(path):
    """Load a .tflite or .keras model and return the backend's native object.

    ponytail: returns the raw interpreter/model, not a wrapper. A unified predict() has to
    normalize tflite's set_tensor/invoke/get_tensor against keras' predict(), which needs the
    lab's model I/O conventions pinned down first. Add the wrapper when those are settled.
    """
    suffix = os.path.splitext(str(path))[1].lower()
    if suffix == TFLITE_SUFFIX:
        return _load_tflite(path)
    if suffix == KERAS_SUFFIX:
        return _load_keras(path)
    raise ValueError(f"Unsupported model format {suffix!r}: expected {TFLITE_SUFFIX} or {KERAS_SUFFIX}")


def _load_tflite(path):
    # tflite_runtime first: it is what the Summit has, and it is far lighter than full TF.
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            import tensorflow as tf
        except ImportError:
            # On 3.7 and 3.10-3.13 the markers guarantee one of these, so getting here means
            # either a marker gap (3.8/3.9, or 3.14+ where TF has no wheel) or an environment
            # assembled some other way (--no-deps, a stripped venv).
            raise ImportError(
                f"Loading {TFLITE_SUFFIX} needs a TFLite interpreter, and neither tflite_runtime nor "
                "tensorflow is installed. `pip install unrbl-ml-pipeline` provides one on "
                "Python 3.7 (unrbl-tflite-runtime) and on 3.10-3.13 (tensorflow), but not on "
                f"3.8/3.9 or 3.14+ (running {sys.version_info[0]}.{sys.version_info[1]}) -- TensorFlow has no wheels there."
            )
        # TensorFlow exposes this through its public lazy API; a direct import
        # from tensorflow.lite is not supported by every installed TF version.
        Interpreter = tf.lite.Interpreter
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def _load_keras(path):
    # Checked before the import so the Summit gets a useful answer instead of "No module named
    # keras", which would imply the problem is fixable with pip. It is not.
    if sys.version_info < (3, 11):
        raise RuntimeError(
            f"The {KERAS_SUFFIX} format requires Keras 3, which needs Python >=3.11 (running {sys.version_info[0]}.{sys.version_info[1]}). "
            f"This machine cannot load {KERAS_SUFFIX} at any version. Convert the model to {TFLITE_SUFFIX} on a modern "
            "machine and deploy that instead."
        )
    try:
        import keras
    except ImportError:
        raise ImportError(
            f"Loading {KERAS_SUFFIX} needs Keras, which a normal `pip install unrbl-ml-pipeline` brings in "
            "via tensorflow on this interpreter. Reinstall without --no-deps."
        )
    return keras.saving.load_model(path)

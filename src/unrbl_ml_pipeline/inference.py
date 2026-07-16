"""Model loading for deployment targets.

Core module: must import on a core-only install (numpy only) and stay Python 3.7-syntax-clean.

Backends are imported lazily *inside* each branch on purpose. A core-only install has neither
tensorflow nor tflite_runtime, and `import unrbl_ml_pipeline` must still work on the Summit.
"""

import os
import sys

TFLITE_SUFFIX = ".tflite"
KERAS_SUFFIX = ".keras"

_SUMMIT_WHEEL_URL = "https://github.com/UtahNeuroroboticsLab/tflite_for_rippleSummit"


def load_model(path):
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
    raise ValueError(
        "Unsupported model format {!r}: expected {} or {}".format(
            suffix, TFLITE_SUFFIX, KERAS_SUFFIX
        )
    )


def _load_tflite(path):
    # tflite_runtime first: it is what the Summit has, and it is far lighter than full TF.
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError:
            raise ImportError(
                "Loading {} needs a TFLite interpreter, and neither tflite_runtime nor "
                "tensorflow is installed.\n"
                "  Ripple Summit (Python 3.7, i686): install the lab-built wheel from {}\n"
                "  Linux x86_64: pip install tflite-runtime\n"
                "  Anywhere else: pip install \"unrbl-ml-pipeline[train]\" (uses tf.lite)".format(
                    TFLITE_SUFFIX, _SUMMIT_WHEEL_URL
                )
            )
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def _load_keras(path):
    # Checked before the import so the Summit gets a useful answer instead of "No module named
    # keras", which would imply the problem is fixable with pip. It is not.
    if sys.version_info < (3, 11):
        raise RuntimeError(
            "The {} format requires Keras 3, which needs Python >=3.11 (running {}.{}). "
            "This machine cannot load {} at any version. Convert the model to {} on a modern "
            "machine and deploy that instead.".format(
                KERAS_SUFFIX,
                sys.version_info[0],
                sys.version_info[1],
                KERAS_SUFFIX,
                TFLITE_SUFFIX,
            )
        )
    try:
        import keras
    except ImportError:
        raise ImportError(
            "Loading {} needs Keras: pip install \"unrbl-ml-pipeline[train]\"".format(
                KERAS_SUFFIX
            )
        )
    return keras.saving.load_model(path)

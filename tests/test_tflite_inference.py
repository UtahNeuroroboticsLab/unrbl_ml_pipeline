"""Load a .tflite through load_model_file(), run it, assert on the result.

Author: Leonardo M. Ferrisi
Date: 2026-07-16

Runs real inference rather than just importing. An import-only check passes even when the
interpreter cannot execute a graph -- which is the failure that would actually matter on the
Summit, and the one thing a bad interpreter build could plausibly break.

This test is the whole marker split, observed from the outside. It exercises whichever backend
the interpreter resolved to and asserts they agree on the answer:
  - on 3.7 (CI's i686 container, and the Summit) -> unrbl-tflite-runtime 2.5.0
  - on newer Python, without that runtime        -> tf.lite.Interpreter

Must stay 3.7-syntax-clean: it runs on the Summit's interpreter in CI.
"""

import os
import sys
from importlib.util import find_spec

import numpy as np
import pytest

from unrbl_ml_pipeline.inference import load_model_file

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy_model.tflite")

# find_spec, not import: importing tensorflow here would cost seconds and, worse, poison
# sys.modules for anything in this session that checks what got imported.
_HAS_INTERPRETER = find_spec("tflite_runtime") is not None or find_spec("tensorflow") is not None

# Newer Python can use TensorFlow without installing the Summit runtime. Skip
# only if neither backend is installed there. Python 3.7 must have a working
# interpreter: a missing Summit runtime should fail its deployment/CI check.
pytestmark = pytest.mark.skipif(
    sys.version_info[:2] != (3, 7) and not _HAS_INTERPRETER,
    reason="neither TensorFlow nor tflite_runtime is installed",
)


def test_tflite_roundtrip():
    interpreter = load_model_file(MODEL)

    # load_model_file returns the raw interpreter, already allocated.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], np.array([[10.0]], dtype=np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])

    assert prediction.shape == (1, 1), f"unexpected output shape: {prediction.shape!r}"

    # The toy model is a single dense layer trained to approximate X -> 3X, so it lands near
    # 30.0 rather than on it.
    assert abs(prediction[0][0] - 30.0) < 0.5, f"expected ~30.0, got {prediction[0][0]!r}"

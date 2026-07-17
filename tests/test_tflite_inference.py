"""End-to-end check: load a .tflite through load_model(), run it, assert on the result.

Runs real inference rather than just importing. An import-only check passes even when the
interpreter cannot execute a graph -- which is the failure that would actually matter on the
Summit, and the one thing a bad interpreter build could plausibly break.

This test is the whole marker split, observed from the outside. It exercises whichever backend
the interpreter resolved to and asserts they agree on the answer:
  - on 3.7 (CI's i686 container, and the Summit) -> unrbl-tflite-runtime 2.5.0
  - on 3.10+                                     -> tensorflow.lite, via load_model's fallback

Must stay 3.7-syntax-clean: it runs on the Summit's interpreter in CI.
"""

import os
from importlib.util import find_spec

import numpy as np
import pytest

from unrbl_ml_pipeline.inference import load_model

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy_model.tflite")

# find_spec, not import: importing tensorflow here would cost seconds and, worse, poison
# sys.modules for anything in this session that checks what got imported.
_HAS_INTERPRETER = find_spec("tflite_runtime") is not None or find_spec("tensorflow") is not None

# Skips only where an interpreter is *impossible*, never where one is merely missing: the
# marker gaps at 3.8/3.9 and 3.14+ (see pyproject.toml). Every interpreter CI actually runs --
# 3.7 i686 and 3.11-3.13 -- has one, so this never skips in CI. If it ever does, the markers
# regressed and that is the bug, not this test.
pytestmark = pytest.mark.skipif(
    not _HAS_INTERPRETER,
    reason="no TFLite interpreter on this Python (marker gap: 3.8/3.9 or 3.14+)",
)


def test_tflite_roundtrip():
    interpreter = load_model(MODEL)

    # load_model returns the raw interpreter, already allocated -- see its ponytail note.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], np.array([[10.0]], dtype=np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])

    assert prediction.shape == (1, 1), "unexpected output shape: %r" % (prediction.shape,)
    # The toy model is a single dense layer trained to approximate X -> 3X, so it lands near
    # 30.0 rather than on it.
    assert abs(prediction[0][0] - 30.0) < 0.5, "expected ~30.0, got %r" % (prediction[0][0],)

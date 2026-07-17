"""Guards the 3.7 boundary.

This is the mechanical enforcement of the import rule. Without it the rule is only a README
paragraph, and the first module-level `import tensorflow` in a root module breaks the Summit
silently -- on a dev box, where the markers always install TF, everything still looks fine.

Now that TF is a core dependency on 3.10+ this matters more, not less: there is no longer any
dev environment where a stray root-level `import tensorflow` fails on its own.
"""

import subprocess
import sys

import pytest

from unrbl_ml_pipeline.inference import load_model


def test_core_import_does_not_pull_tensorflow():
    # Subprocess, not a plain assert: this test session may already have imported tensorflow
    # via some other test, which would mask a module-level import in the core package.
    code = (
        "import unrbl_ml_pipeline, unrbl_ml_pipeline.inference, sys; "
        "heavy = [m for m in ('tensorflow', 'keras', 'tflite_runtime') if m in sys.modules]; "
        "assert not heavy, 'core imported at module level: %s' % heavy"
    )
    subprocess.check_call([sys.executable, "-c", code])


def test_load_model_rejects_unknown_suffix():
    with pytest.raises(ValueError):
        load_model("model.h5")

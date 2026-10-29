"""Post-install sanity check: the package imports, TF imports, and (if present) the GPU computes.

Hard failure (exit 1): unrbl_ml_pipeline or tensorflow does not import.
Soft warning (exit 0): no GPU visible - CPU-only is a valid dev setup.
"""
import sys

import unrbl_ml_pipeline

print("unrbl_ml_pipeline", unrbl_ml_pipeline.__version__)

try:
    import tensorflow as tf
except ImportError as exec:
    print(f"FAIL: tensorflow does not import: {exec}")
    sys.exit(1)

print("tensorflow", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    print("WARNING: no GPU visible to TensorFlow; running on CPU only.")
    sys.exit(0)

for gpu in gpus:
    print("GPU:", gpu.name)

with tf.device("/GPU:0"):
    out = tf.matmul(tf.ones((64, 64)), tf.ones((64, 64)))
if float(out[0, 0]) != 64.0:
    print("FAIL: GPU matmul returned the wrong result")
    sys.exit(1)
print("OK: GPU matmul works")

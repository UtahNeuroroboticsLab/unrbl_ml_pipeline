"""Bounded transfers from NumPy arrays in host memory to TensorFlow models."""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def batched_dataset(x, y=None, *, batch_size, shuffle=False, seed=42,
                    sample_weight=None):
    """Yield CPU batches without converting an entire split to a tensor.

    Arrays remain in host memory. Only integer indices are shuffled; validation
    and prediction preserve sample order, including the final partial batch.
    One batch is prefetched. Each new training iteration uses a new seeded
    permutation, so finite datasets can be reused across Keras epochs.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if shuffle not in (False, True, "batch"):
        raise ValueError("shuffle must be False, True, or 'batch'")
    if sample_weight is not None and y is None:
        raise ValueError("sample_weight requires targets")
    arrays = (x,) if y is None else (x, y)
    if sample_weight is not None:
        arrays += (sample_weight,)
    count = len(x)
    if not count or any(len(array) != count for array in arrays):
        raise ValueError("Inputs, targets, and weights must have matching nonzero lengths")
    signature = tuple(tf.TensorSpec((None,) + tuple(array.shape[1:]),
                                    tf.as_dtype(array.dtype)) for array in arrays)
    rng = np.random.default_rng(seed)

    def batches():
        indices = rng.permutation(count) if shuffle is True else None
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            if indices is not None:
                selection = indices[start:end]
            elif shuffle == "batch":
                selection = rng.permutation(np.arange(start, end))
            else:
                selection = slice(start, end)
            yield tuple(np.ascontiguousarray(array[selection]) for array in arrays)

    # from_tensor_slices/from_tensors would first convert the full arrays. The
    # generator instead closes over NumPy references and converts only batches.
    with tf.device("/CPU:0"):
        dataset = tf.data.Dataset.from_generator(batches, output_signature=signature)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(
            (count + batch_size - 1) // batch_size,
        ))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        dataset = dataset.with_options(options).prefetch(1)
    return dataset

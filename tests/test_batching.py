"""Regression checks for bounded CPU-to-device transfers."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from unrbl_ml_pipeline.training.batching import batched_dataset


class SliceOnlyArray:
    """Fail if a data adapter tries to convert the entire host array."""

    def __init__(self, values):
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.read_sizes = []

    def __len__(self):
        return len(self.values)

    def __array__(self, *args, **kwargs):
        raise AssertionError("The entire split must not be converted to a tensor")

    def __getitem__(self, selection):
        batch = self.values[selection]
        self.read_sizes.append(len(batch))
        return batch


def test_batches_stay_on_cpu_and_keep_remainder_and_alignment():
    x = SliceOnlyArray(np.arange(55, dtype=np.float32).reshape(11, 5))
    y = SliceOnlyArray(np.arange(11, dtype=np.float32)[:, None])
    weights = SliceOnlyArray(np.arange(11, dtype=np.float32) + 1)
    # Even a caller's GPU context must not place the full arrays on that device.
    with tf.device("/GPU:0"):
        dataset = batched_dataset(x, y, sample_weight=weights, batch_size=4)
    assert int(dataset.cardinality()) == 3
    batches = list(dataset)
    assert [len(batch[0]) for batch in batches] == [4, 4, 3]
    assert all("CPU:0" in tensor.device for batch in batches for tensor in batch)
    for array, position in ((x, 0), (y, 1), (weights, 2)):
        np.testing.assert_array_equal(np.concatenate([b[position].numpy() for b in batches]), array.values)
        assert max(array.read_sizes) <= 4


def test_seeded_shuffle_changes_each_epoch_without_losing_samples():
    x = np.arange(23, dtype=np.float32)[:, None]
    y = x * 2
    dataset = batched_dataset(x, y, batch_size=6, shuffle=True, seed=7)
    duplicate = batched_dataset(x, y, batch_size=6, shuffle=True, seed=7)

    def read_epoch(data):
        batches = list(data.as_numpy_iterator())
        features = np.concatenate([b[0] for b in batches])
        targets = np.concatenate([b[1] for b in batches])
        np.testing.assert_array_equal(targets, features * 2)
        np.testing.assert_array_equal(np.sort(features[:, 0]), x[:, 0])
        return features

    first = read_epoch(dataset)
    second = read_epoch(dataset)
    np.testing.assert_array_equal(first, read_epoch(duplicate))
    np.testing.assert_array_equal(second, read_epoch(duplicate))
    assert not np.array_equal(first, second)


def test_large_logical_split_reads_only_requested_batches():
    # Same shape as the user's failing validation split, backed by one window.
    virtual = np.broadcast_to(np.zeros((1, 10, 496), dtype=np.float32), (54186, 10, 496))
    x = SliceOnlyArray(virtual)
    dataset = batched_dataset(x, batch_size=128)
    batches = list(dataset.take(2))
    assert len(batches) == 2
    assert all(batch[0].shape == (128, 10, 496) for batch in batches)
    assert max(x.read_sizes) == 128
    # The bounded prefetch may read ahead, but never scans the whole split.
    assert sum(x.read_sizes) < len(x)


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0}, {"batch_size": 4, "shuffle": "invalid"},
    {"batch_size": 4, "sample_weight": np.ones(3)},
])
def test_invalid_batching_arguments(kwargs):
    with pytest.raises(ValueError):
        batched_dataset(np.ones((5, 2), dtype=np.float32), **kwargs)

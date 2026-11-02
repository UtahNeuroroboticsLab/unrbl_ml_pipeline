"""Reference AdamW update geometry, including the two LSTM bias parameters."""
import numpy as np
import tensorflow as tf

from unrbl_ml_pipeline.training.mamba_layers import ReferenceAdamW, ReferenceLSTMCell


def test_corrected_moment_epsilon_and_decay_exclusion():
    ordinary = tf.Variable([1.0, -1.0])
    special = tf.Variable([1.0, -1.0])
    opt = ReferenceAdamW(learning_rate=.0005, weight_decay=.0001)
    opt.exclude_from_weight_decay(var_list=[special])
    gradient = tf.constant([1e-9, -1e-6])
    opt.apply_gradients([(gradient, ordinary), (gradient, special)])
    update = .0005*gradient.numpy()/(np.abs(gradient.numpy())+1e-8)
    np.testing.assert_allclose(special.numpy(), np.array([1., -1.])-update, atol=1e-7)
    np.testing.assert_allclose(ordinary.numpy(), np.array([1., -1.])*(1-.0005*.0001)-update, atol=1e-7)


def test_lstm_biases_receive_separate_gradients():
    cell = ReferenceLSTMCell(3)
    x = tf.ones((2, 4))
    with tf.GradientTape() as tape:
        y, _ = cell(x, (tf.zeros((2, 3)), tf.zeros((2, 3))))
        loss = tf.reduce_sum(y*y)
    left, right = tape.gradient(loss, (cell.bias, cell.recurrent_bias))
    assert cell.bias is not cell.recurrent_bias
    np.testing.assert_array_equal(left, right)
    assert np.linalg.norm(left.numpy()) > 0

"""Frozen PyTorch oracle parity; no PyTorch or reference tree needed to run."""

from pathlib import Path

import numpy as np
import pytest

from unrbl_ml_pipeline import models

KINDS = ("mamba", "tcn_mamba", "mamba_tcn", "mamba_tcl", "mamba_lstm")


def assign_reference(model, fixture):
    def w(key):
        return fixture["weight." + key]

    def dense(layer, key):
        values = [w(key + ".weight").T]
        if layer.use_bias:
            values.append(w(key + ".bias"))
        layer.set_weights(values)

    def norm(layer, key):
        layer.set_weights([w(key + ".weight"), w(key + ".bias")])

    dense(model.project, "project")
    norm(model.mamba.norm, "mamba.layer.norm")
    norm(model.mamba.final_norm, "mamba.norm_f")
    root = "mamba.layer.mixer."
    for name in ("in_proj", "x_proj", "dt_proj", "out_proj"):
        dense(getattr(model.mamba, name), root + name)
    model.mamba.conv_w.assign(w(root + "conv1d.weight")[:, 0, :].T)
    model.mamba.conv_b.assign(w(root + "conv1d.bias"))
    model.mamba.A_log.assign(w(root + "A_log"))
    model.mamba.D.assign(w(root + "D"))
    if model.has_tcn:
        for i, layer in enumerate(model.convs):
            name = f"tcn.network.{i // 2}.conv{i % 2 + 1}"
            layer.v.assign(w(name + ".weight_v").transpose(2, 1, 0))
            layer.g.assign(w(name + ".weight_g").reshape(-1))
            layer.bias.assign(w(name + ".bias"))
        if model.skip is not None:
            model.skip.set_weights(
                [
                    w("tcn.network.0.downsample.weight")[:, :, 0].T,
                    w("tcn.network.0.downsample.bias"),
                ]
            )
    if model.has_lstm:
        norm(model.recurrent_norm, "recurrent_norm")
        model.lstm.cell.set_weights(
            [
                w("lstm.weight_ih_l0").T,
                w("lstm.weight_hh_l0").T,
                w("lstm.bias_ih_l0"),
                w("lstm.bias_hh_l0"),
            ]
        )
    norm(model.feature_norm, "feature_norm")
    norm(model.fusion_norm, "fusion_norm")
    dense(model.fc, "regression_fc")
    dense(model.head, "regression_head")


def oracle_state(fixture, name):
    return (
        fixture[name + ".0"].transpose(0, 2, 1),
        fixture[name + ".1"],
        fixture[name + ".2"],
    )


@pytest.mark.parametrize("kind", KINDS)
def test_pinned_reference_predictions(kind):
    fixture = np.load(Path(__file__).parent / "fixtures/mamba" / f"{kind}.npz")
    model = models.get(kind + "_decoder").build(
        (10, 5), 2, architecture={"mamba_width": 8, "tcn_width": 8}
    )
    assign_reference(model, fixture)
    x = fixture["x"]
    np.testing.assert_allclose(
        model.predict_windows(x[:, :10]), fixture["reset"], atol=3e-6, rtol=3e-5
    )
    initial = model.prefill(x[:, :9])
    for a, b in zip(initial, oracle_state(fixture, "initial")):
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-5)
    for start, expected, end in [
        (initial, "carry", "end"),
        (oracle_state(fixture, "nonzero_initial"), "nonzero", "nonzero_end"),
    ]:
        pred, state = model.step(x[:, 9:], start)
        np.testing.assert_allclose(pred, fixture[expected], atol=3e-6, rtol=3e-5)
        for a, b in zip(state, oracle_state(fixture, end)):
            np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_reference_second_derivatives(kind):
    import tensorflow as tf

    fixture = np.load(Path(__file__).parent / "fixtures/mamba" / f"{kind}.npz")
    model = models.get(kind + "_decoder").build(
        (10, 5), 2, architecture={"mamba_width": 8, "tcn_width": 8}
    )
    assign_reference(model, fixture)
    z = tf.constant(fixture["x"][:, 9:10])
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            inner.watch(z)
            pred = model.step(z, oracle_state(fixture, "nonzero_initial"))[0][:, 0]
            scalar = tf.reduce_sum(pred)
        gradient = inner.gradient(scalar, z)
        penalty = tf.reduce_mean(tf.reduce_sum(gradient**2, (1, 2))) / 2
    pg = outer.gradient(penalty, model.project.kernel)
    np.testing.assert_allclose(penalty, fixture["jacobian_value"], atol=2e-7, rtol=1e-4)
    np.testing.assert_allclose(
        pg, fixture["jacobian_project_gradient"].T, atol=1e-5, rtol=1e-3
    )

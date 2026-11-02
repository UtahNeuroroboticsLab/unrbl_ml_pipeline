"""Keras port of the reference official_v2 mamba_lstm regression decoder."""

from ._mamba_defaults import OBJECTIVES, build_decoder, defaults

name = "mamba_lstm_decoder"
description = "Mamba-1 mamba_lstm decoder with carry/reset training"
paper = "Mamba-1; decoder-design official_v2 adapter"
model_type = "keras"
training_route = "mamba"
supported_objectives = {"carry": OBJECTIVES, "reset": OBJECTIVES}
default_config = defaults()


def build(input_shape, num_classes, norm_stats=None, **config):
    return build_decoder("mamba_lstm", input_shape, num_classes, norm_stats, **config)

# Source attribution

Mamba-1 equations and initialization are ported from state-spaces/mamba, commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`.

Copyright (c) 2023, Tri Dao, Albert Gu. Apache License 2.0; see MAMBA_LICENSE.

TCN topology and weight normalization follow locuslab/TCN, commit `2f8c2b817050206397458dfd1f5a25ce8a32fe65`. See TCN_LICENSE for its copyright and MIT license.

The EMG projections, hybrid readouts and regression topology follow the local decoder-design official_v2 adapters. This package contains a TensorFlow/Keras port, not a runtime dependency on the reference checkout. The development oracle manifest in tests/fixtures/mamba/manifest.json records source hashes and fixture provenance.

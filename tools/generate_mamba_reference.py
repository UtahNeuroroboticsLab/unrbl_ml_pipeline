"""Generate frozen oracle fixtures with isolated CPU PyTorch, never at package runtime.

Usage: /tmp/unrbl-mamba-parity/bin/python tools/generate_mamba_reference.py
Source: .ref/decoder-design/tcn_lstm_cls official_v2, pinned author provenance.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / ".ref/decoder-design/tcn_lstm_cls"
sys.path.insert(0, str(REF))
from official_models import OfficialDecoder

torch.set_num_threads(1)
out = ROOT / "tests/fixtures/mamba"
out.mkdir(parents=True, exist_ok=True)
for kind in ("mamba", "tcn_mamba", "mamba_tcn", "mamba_tcl", "mamba_lstm"):
    torch.manual_seed(83)
    model = OfficialDecoder(
        kind,
        num_features=5,
        num_outputs=2,
        mamba_width=8,
        tcn_width=8,
        state_mode="carry",
        dropout=0.2,
    ).eval()
    x = torch.randn(2, 20, 5)
    with torch.no_grad():
        initial = model.prefill(x[:, :9])
        p, _, end = model.recurrent(x[:, 9:], initial)
        # A deliberately nonzero supplied recurrent state tests the entire state contract.
        nonzero = [v + 0.1 * torch.randn_like(v) for v in initial]
        pn, _, sn = model.recurrent(x[:, 9:], nonzero)
        model.stateful = False
        reset = model(x[:, :10])[0][:, 0]
    arrays = {
        "x": x.numpy(),
        "carry": p.numpy(),
        "reset": reset.numpy(),
        "nonzero": pn.numpy(),
    }
    # Double-backward oracle for a conditional newest-sample Jacobian.
    z = x[:, 9:10].detach().requires_grad_(True)
    with torch.backends.mkldnn.flags(enabled=False):
        pred = model.recurrent(z, [v.detach() for v in nonzero])[0][:, 0]
        gradient = torch.autograd.grad(pred.sum(), z, create_graph=True)[0]
        penalty = gradient.square().sum((1, 2)).mean() / 2
        pg = torch.autograd.grad(penalty, model.project.weight)[0]
    arrays["jacobian_value"] = penalty.detach().numpy()
    arrays["jacobian_project_gradient"] = pg.detach().numpy()
    arrays.update(
        {"weight." + k: v.detach().numpy() for k, v in model.state_dict().items()}
    )
    for label, values in [
        ("initial", initial),
        ("end", end),
        ("nonzero_initial", nonzero),
        ("nonzero_end", sn),
    ]:
        arrays.update({label + "." + str(i): v.numpy() for i, v in enumerate(values)})
    np.savez_compressed(out / (kind + ".npz"), **arrays)
manifest = {
    "reference": "decoder-design official_v2",
    "torch_version": torch.__version__,
    "seed": 83,
    "author_provenance": json.loads((REF / "official_provenance.json").read_text()),
    "adapter_sha256": {
        name: hashlib.sha256((REF / name).read_bytes()).hexdigest()
        for name in ("official_models.py", "official_backbones.py")
    },
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

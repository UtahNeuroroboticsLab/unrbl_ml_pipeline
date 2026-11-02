"""Audit local clocks and run bounded CPU real-data integration checks."""

import json
import resource
import time
from pathlib import Path

import numpy as np

from unrbl_ml_pipeline.training import resolve_config, run_training
from unrbl_ml_pipeline.training.preprocessing import run_pipeline

output = Path("outputs/mamba_validation")
output.mkdir(parents=True, exist_ok=True)
config = resolve_config("mamba_decoder", config_path="config_mamba.jsonc")
legacy = dict(config, data=dict(config["data"], require_usable_timestamps=False))
base = run_pipeline(["load_kdf", "quality_filter"], legacy)
current = run_pipeline(["load_kdf", "quality_filter"], config)
audit = {
    "matched": len(base.selection_report),
    "beta_qualified": len(base.files),
    "mamba_eligible": len(current.files),
    "selection": current.selection_report,
    "clock_checks": [
        {
            "path": f.path,
            "nominal_interval": f.nominal_interval,
            "nonpositive_deltas": int(
                (np.diff(np.asarray(f.nip_time, dtype=float)) <= 0).sum()
            ),
        }
        for f in base.files
    ],
}
(output / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
results = []
for mode, loss in [("carry", "mse"), ("reset", "mse"), ("carry", "mse_pinn_jacobian")]:
    start = time.perf_counter()
    _, result = run_training(
        "mamba_decoder",
        config_path="config_mamba.jsonc",
        config={
            "data": {"kdf_glob": "hook*/trainKDF_202607*.kdf"},
            "epochs": 1,
            "state_mode": mode,
            "architecture": {
                "mamba_width": 8,
                "state_size": 4,
                "tcn_width": 8,
                "lstm_units": 8,
                "head_units": 8,
            },
            "loss": {"type": loss, "warmup_epochs": 0},
        },
        output_dir=output / "real_runs",
        verbose=0,
    )
    record = {
        "mode": mode,
        "loss": loss,
        "seconds": time.perf_counter() - start,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "run_dir": str(result["run_dir"]),
        "test_loss": result["test_loss"],
    }
    results.append(record)
    print(json.dumps(record), flush=True)
(output / "real_smoke.json").write_text(json.dumps(results, indent=2) + "\n")

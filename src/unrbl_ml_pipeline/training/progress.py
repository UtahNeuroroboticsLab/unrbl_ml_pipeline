"""Console progress shared by the direct training API and recorded runs."""
from __future__ import annotations

import os


class TrainingProgress:
    """Number the stages that will actually run; flush messages before work."""

    def __init__(self, steps, *, save_model=False, reports=False, verbose=True):
        self.total = len(steps) + 3 + int(save_model) + int(reports)
        self.current = 0
        self.enabled = bool(verbose)

    def message(self, text):
        if self.enabled:
            print(text, flush=True)

    def stage(self, description):
        self.current += 1
        self.message(f"\n=== [{self.current}/{self.total}] {description} ===")

    def device(self, model_type):
        if not self.enabled:
            return
        if model_type == "estimator":
            self.message("\n=== Estimator training on CPU ===")
            return
        import tensorflow as tf
        gpus = tf.config.get_visible_devices("GPU")
        if gpus:
            self.message(f"\n=== GPU detected: {', '.join(gpu.name for gpu in gpus)} ===")
        else:
            self.message("\n=== No GPU detected; training on CPU ===")

    def file(self, path, config, *, decision, score=None, threshold=None,
             samples=None, reason=None):
        base_dir = config.get("data", {}).get("base_dir")
        name = os.path.relpath(path, base_dir) if base_dir else os.path.basename(path)
        if score is None:
            tag = decision
        elif decision == "KEEP":
            tag = f"KEEP   score={score:.3f}"
        else:
            tag = f"REJECT score={score:.3f} < {threshold:g}"
        detail = f"  T={samples}" if samples is not None else ""
        if reason:
            detail += f": {reason}"
        self.message(f"  [{tag}] {name}{detail}")

    def preprocessing_stage(self, name, config):
        descriptions = {
            "load_kdf": "Loading .kdf files",
            "quality_filter": f"Scoring recordings (beta={float(config.get('data', {}).get('beta_threshold', 0)):g})",
            "normalize": "Computing normalization statistics from TRAIN split",
            "window": f"Building windows (window_size={config.get('model', {}).get('window_size', 50)})",
        }
        if name.startswith("split_"):
            description = f"Splitting recordings (method={name})"
        else:
            description = descriptions.get(name, f"Preprocessing: {name}")
        self.stage(description)

    def preprocessing_result(self, name, state):
        if name == "load_kdf":
            self.message(f"  Matched {len(state.selection_report)} recordings; loaded {len(state.files)}.")
        elif name == "quality_filter":
            rejected = sum(r["status"] == "rejected" for r in state.selection_report)
            skipped = sum(r["status"] in ("excluded", "load_error") for r in state.selection_report)
            self.message(f"  Kept {len(state.files)} recordings; rejected {rejected}; skipped {skipped}.")
        elif name.startswith("split_"):
            info = state.split_info
            if "n_detected_trials" in info:
                self.message(f"  Detected trials: {info['n_detected_trials']} across {len(state.files)} recordings.")
            if "n_movement_types" in info:
                self.message(f"  Movement types: {info['n_movement_types']}")
            counts = " ".join(f"{split}={len(getattr(state, split + '_files'))}"
                              for split in ("train", "val", "test"))
            self.message(f"  Split segments: {counts}")
        elif name == "window":
            shapes = " ".join(f"{split}={getattr(state, 'X_' + split).shape}"
                              for split in ("train", "val", "test")
                              if getattr(state, 'X_' + split) is not None)
            self.message(f"  Windows: {shapes}")

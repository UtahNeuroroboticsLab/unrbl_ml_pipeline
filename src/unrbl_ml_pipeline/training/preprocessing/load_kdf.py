"""Preprocessing step: load .kdf binary files.

Author: Leonardo M. Ferrisi
Date: 2026-10-29
"""
from __future__ import annotations

import glob as glob_mod
import os
import re

import numpy as np

from ._types import KdfFile

# --- KEF trial event parsing ---

class TrialEvent:
    __slots__ = ("mvnt_mat", "targ_on_ts", "trial_ts")
    def __init__(self, targ_on_ts, trial_ts, mvnt_mat):
        self.targ_on_ts = targ_on_ts
        self.trial_ts = trial_ts
        self.mvnt_mat = mvnt_mat


def _parse_scalar(text):
    return float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.strip())[0])


def _matlab_array(text):
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    rows = []
    for row in re.split(r";|\n", s):
        row = row.strip()
        if not row:
            continue
        rows.append([float(x) for x in re.split(r"[,\s]+", row) if x])
    return np.array(rows, dtype=np.float64)


def parse_kef(path):
    """Parse a sibling .kef trial/event file. Returns [] if missing."""
    if not os.path.exists(path):
        return []
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    trials = []
    for chunk in re.split(r"\r?\n", text):
        if not chunk.strip():
            continue
        targ = re.search(r"SS\.TargOnTS\s*=\s*([^;]+);?", chunk)
        trial = re.search(r"SS\.TrialTS\s*=\s*([^;]+);?", chunk)
        mvnt = re.search(r"SS\.MvntMat\s*=\s*(\[[^\]]*\])", chunk)
        if targ is None or trial is None or mvnt is None:
            continue
        try:
            trials.append(TrialEvent(
                targ_on_ts=_parse_scalar(targ.group(1)),
                trial_ts=_parse_scalar(trial.group(1)),
                mvnt_mat=_matlab_array(mvnt.group(1)),
            ))
        except Exception:
            continue
    return trials


def trial_id_from_kef(nip_time, trials):
    """Build per-sample trial IDs from KEF events."""
    n = nip_time.shape[0]
    ids = np.zeros(n, dtype=np.int64)
    if n == 0 or not trials:
        return ids
    for i, tr in enumerate(trials, start=1):
        mask = (nip_time >= tr.targ_on_ts) & (nip_time <= tr.trial_ts)
        ids[mask] = i
    return ids


# --- KDF binary reading ---

def read_kdf_cob(path):
    """Read a COB-variant .kdf. Returns (features [T,F], kinematics [T,K], nip_time [T])."""
    with open(path, "rb") as f:
        raw = np.fromfile(f, dtype="<f4")
    _n_nip, n_feat, n_kin = raw[0:3].astype(int)
    header_len = 3
    n_rows = 1 + n_feat + n_kin
    n_cols = (raw.size - header_len) // n_rows
    if n_cols * n_rows != raw.size - header_len:
        raise ValueError(
            f"{path}: payload size {raw.size - header_len} not a multiple of row count {n_rows}"
        )
    payload = raw[header_len:].reshape((n_rows, n_cols), order="F")
    nip_time = payload[0, :]
    features = payload[1:1 + n_feat, :].T
    kinematics = payload[1 + n_feat:, :].T
    return (
        np.ascontiguousarray(features),
        np.ascontiguousarray(kinematics),
        np.ascontiguousarray(nip_time),
    )


def load_one_kdf(path, kinematic_column_indices=None):
    """Load a single .kdf file into a KdfFile."""
    features, kinematics, nip_time = read_kdf_cob(path)
    if kinematic_column_indices is not None:
        kinematics = kinematics[:, kinematic_column_indices]
    return KdfFile(path=path, features=features, kinematics=kinematics, nip_time=nip_time)


def find_kdf_files(glob_pattern, base_dir=None):
    """Resolve a glob into a sorted list of .kdf paths."""
    if base_dir is not None and not os.path.isabs(glob_pattern):
        glob_pattern = os.path.join(base_dir, glob_pattern)
    raw = glob_mod.glob(glob_pattern, recursive=True)
    return sorted(p for p in raw if os.path.isfile(p) and p.lower().endswith(".kdf"))


# --- Step entry point ---

def run(state, config):
    """Load all .kdf files matching the configured glob pattern.

    Config keys used:
        data.kdf_glob       — glob pattern for .kdf files
        data.base_dir       — base directory (optional)
        data.kinematic_column_indices — column selection (optional)
    """
    data_cfg = config.get("data", {})
    kdf_glob = data_cfg.get("kdf_glob", "**/*.kdf")
    base_dir = data_cfg.get("base_dir", None)
    kin_cols = data_cfg.get("kinematic_column_indices", None)

    paths = find_kdf_files(kdf_glob, base_dir=base_dir)
    if not paths:
        raise FileNotFoundError(
            f"No .kdf files matched pattern '{kdf_glob}'."
        )

    files = []
    for p in paths:
        try:
            files.append(load_one_kdf(p, kinematic_column_indices=kin_cols))
        except Exception as e:
            state.rejected.append((p, str(e)))

    state.files = files
    return state

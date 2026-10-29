#!/usr/bin/env bash
# Build the dev env (.venv, Python 3.12) with CUDA TensorFlow if possible, then verify it.
# uv sync has no post-install hook, so the check is chained here instead.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=3.12

if ! uv sync --python "$PY" --extra dev --extra cuda; then
    echo "WARNING: CUDA extra failed to install; falling back to CPU-only TensorFlow." >&2
    uv sync --python "$PY" --extra dev
fi

uv run python scripts/check_install.py

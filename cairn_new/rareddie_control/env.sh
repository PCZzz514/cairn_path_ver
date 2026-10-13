#!/usr/bin/env bash
# Source only. Runtime for the isolated rareddie_control arms. The cairn
# environment, cairn_dev, accelerated_four, cairn_dsn and scripts/ are
# read-only; caches/temporary files stay under rareddie_control and its logs.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROL_DIR="$ROOT_DIR/rareddie_control"
LOG_DIR="$ROOT_DIR/logs/rareddie_control"
cd "$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1
# rareddie_control (new) and accelerated_four (reused, read-only) resolve from ROOT.
export PYTHONPATH="$ROOT_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TMPDIR="$LOG_DIR/tmp"
export XDG_CACHE_HOME="$CONTROL_DIR/.cache"
export TORCH_HOME="$CONTROL_DIR/.cache/torch"
export TRITON_CACHE_DIR="$CONTROL_DIR/.cache/triton"
export CUDA_CACHE_PATH="$CONTROL_DIR/.cache/cuda"
export MPLCONFIGDIR="$CONTROL_DIR/.cache/matplotlib"
export NUMBA_CACHE_DIR="$CONTROL_DIR/.cache/numba"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON_BIN="$ROOT_DIR/../cairn_dev/.conda/cairn/bin/python"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

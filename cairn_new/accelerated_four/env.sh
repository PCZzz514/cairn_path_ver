#!/usr/bin/env bash
# Source only. Runtime for the isolated accelerated_four replicas. The cairn
# environment, cairn_dev, the original cairn_dsn package and scripts/ are
# read-only; caches/temporary files stay under accelerated_four and its logs.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ACCEL_DIR="$ROOT_DIR/accelerated_four"
LOG_DIR="$ROOT_DIR/logs/accelerated_four"
cd "$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1
# accelerated_four.dsn (new) and cairn_dsn (read-only OLD reference) both resolve from ROOT.
export PYTHONPATH="$ROOT_DIR"
# nvidia-smi index order; identical UUID mapping to the original jobs' GPU4-7.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TMPDIR="$LOG_DIR/tmp"
export XDG_CACHE_HOME="$ACCEL_DIR/.cache"
export TORCH_HOME="$ACCEL_DIR/.cache/torch"
export TRITON_CACHE_DIR="$ACCEL_DIR/.cache/triton"
export CUDA_CACHE_PATH="$ACCEL_DIR/.cache/cuda"
export MPLCONFIGDIR="$ACCEL_DIR/.cache/matplotlib"
export NUMBA_CACHE_DIR="$ACCEL_DIR/.cache/numba"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON_BIN="$ROOT_DIR/../cairn_dev/.conda/cairn/bin/python"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

#!/usr/bin/env bash
# Source only. Existing read-only cairn environment; every cache/temporary file
# stays under crt_kpaths/ and logs/crt_kpaths/.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CRT_DIR="$ROOT_DIR/crt_kpaths"
LOG_DIR="$ROOT_DIR/logs/crt_kpaths"
cd "$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TMPDIR="$LOG_DIR/tmp"
export XDG_CACHE_HOME="$CRT_DIR/.cache"
export TORCH_HOME="$CRT_DIR/.cache/torch"
export TRITON_CACHE_DIR="$CRT_DIR/.cache/triton"
export CUDA_CACHE_PATH="$CRT_DIR/.cache/cuda"
export MPLCONFIGDIR="$CRT_DIR/.cache/matplotlib"
export NUMBA_CACHE_DIR="$CRT_DIR/.cache/numba"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON_BIN="$ROOT_DIR/../cairn_dev/.conda/cairn/bin/python"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

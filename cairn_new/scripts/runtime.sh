#!/usr/bin/env bash
# Source this from this workspace; the existing cairn environment is read-only.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$ROOT_DIR/tmp"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
export TORCH_HOME="$ROOT_DIR/.cache/torch"
export TRITON_CACHE_DIR="$ROOT_DIR/.cache/triton"
export CUDA_CACHE_PATH="$ROOT_DIR/.cache/cuda"
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export NUMBA_CACHE_DIR="$ROOT_DIR/.cache/numba"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON_BIN="$ROOT_DIR/../cairn_dev/.conda/cairn/bin/python"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$HF_HOME" "$NUMBA_CACHE_DIR" "$ROOT_DIR/logs"

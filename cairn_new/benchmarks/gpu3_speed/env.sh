#!/usr/bin/env bash
# Source only. GPU3-only runtime for the isolated speed benchmark. The cairn
# environment, cairn_dev and the original cairn_dsn package are read-only;
# inherited proxy variables are left untouched.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/benchmarks/gpu3_speed"
LOG_DIR="$ROOT_DIR/logs/gpu3_speed"
cd "$ROOT_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
export PYTHONDONTWRITEBYTECODE=1
# dsn_gpu3 (new copy) first; the workspace root only for the read-only OLD reference.
export PYTHONPATH="$BENCH_DIR:$ROOT_DIR"
export TMPDIR="$LOG_DIR/tmp"
export XDG_CACHE_HOME="$BENCH_DIR/.cache"
export TORCH_HOME="$BENCH_DIR/.cache/torch"
export TRITON_CACHE_DIR="$BENCH_DIR/.cache/triton"
export CUDA_CACHE_PATH="$BENCH_DIR/.cache/cuda"
export MPLCONFIGDIR="$BENCH_DIR/.cache/matplotlib"
export NUMBA_CACHE_DIR="$BENCH_DIR/.cache/numba"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON_BIN="$ROOT_DIR/../cairn_dev/.conda/cairn/bin/python"
GPU3_UUID="$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader)"
export GPU3_UUID
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

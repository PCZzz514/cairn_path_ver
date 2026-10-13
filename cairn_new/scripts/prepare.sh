#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
export CUDA_VISIBLE_DEVICES=''
exec "$PYTHON_BIN" -m cairn_dsn.prepare "$@"

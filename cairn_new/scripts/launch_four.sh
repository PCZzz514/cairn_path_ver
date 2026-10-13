#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
if [[ "${1:-}" != --launch-reviewed || $# != 1 ]]; then
  echo 'Usage: bash scripts/launch_four.sh --launch-reviewed (only after coordinator review)' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=''
exec "$PYTHON_BIN" scripts/server_jobs.py --launch-reviewed

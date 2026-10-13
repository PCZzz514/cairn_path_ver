#!/usr/bin/env bash
# Launch exactly one reviewed arm. No training starts by sourcing runtime.sh.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
VARIANT="${1:?variant: path or no_path}"
PROTOCOL="${2:?protocol: single_drugout or double_drugout}"
shift 2
case "$VARIANT/$PROTOCOL" in
  path/single_drugout) export CUDA_VISIBLE_DEVICES=4 ;;
  path/double_drugout) export CUDA_VISIBLE_DEVICES=5 ;;
  no_path/single_drugout) export CUDA_VISIBLE_DEVICES=6 ;;
  no_path/double_drugout) export CUDA_VISIBLE_DEVICES=7 ;;
  *) echo 'Expected path|no_path and single_drugout|double_drugout' >&2; exit 2 ;;
esac
exec "$PYTHON_BIN" -m cairn_dsn.train --config "configs/${VARIANT}_${PROTOCOL}.yaml" "$@"

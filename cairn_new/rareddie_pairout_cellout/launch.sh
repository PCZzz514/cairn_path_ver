#!/usr/bin/env bash
# Host launch of both rareddie_pairout_cellout arms (GPU6 pairout, GPU7
# cellout). Codex runs this ONLY after writing
# logs/rareddie_pairout_cellout/review_approval.json; the controller refuses otherwise.
# Each arm: detached session (wrapper orphaned -> PPID 1) holding its job.lock,
# inherited by the training child, --fresh. Then waits (bounded) for the actual
# first update and a 100-update line of both arms; writes launch.json.
# Usage: bash rareddie_pairout_cellout/launch.sh --launch-reviewed
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ "${1:-}" != --launch-reviewed || $# != 1 ]]; then
  echo 'Usage: bash rareddie_pairout_cellout/launch.sh --launch-reviewed (only after Codex review approval)' >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" rareddie_pairout_cellout/jobs.py --launch-reviewed
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" rareddie_pairout_cellout/confirm_launch.py --wait 1800

#!/usr/bin/env bash
# Host launch of both pairout_cellout full-path arms (GPU6 pairout, GPU7
# cellout; same-DDI-split seed42 replicates). Codex runs this ONLY after writing
# logs/pairout_cellout/review_approval.json; the controller refuses otherwise.
# Each arm: detached session (wrapper orphaned -> PPID 1) holding its job.lock,
# inherited by the training child, --fresh. Then waits (bounded) for the actual
# first update and a 100-update line of both arms; writes launch.json.
# Usage: bash pairout_cellout/launch.sh --launch-reviewed
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ "${1:-}" != --launch-reviewed || $# != 1 ]]; then
  echo 'Usage: bash pairout_cellout/launch.sh --launch-reviewed (only after Codex review approval)' >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" pairout_cellout/jobs.py --launch-reviewed
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" pairout_cellout/confirm_launch.py --wait 1800

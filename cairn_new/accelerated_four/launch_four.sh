#!/usr/bin/env bash
# Host launch of all four accelerated replicas. Codex runs this ONLY after
# writing logs/accelerated_four/review_approval.json; the controller refuses
# otherwise. Each arm: detached session (wrapper orphaned -> PPID 1) holding
# its job.lock, inherited by the training child. Then waits (bounded) for real
# progress of every arm and records logs/accelerated_four/launch.json.
# Usage: bash accelerated_four/launch_four.sh --launch-reviewed
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ "${1:-}" != --launch-reviewed || $# != 1 ]]; then
  echo 'Usage: bash accelerated_four/launch_four.sh --launch-reviewed (only after Codex review approval)' >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" accelerated_four/jobs.py --launch-reviewed
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" accelerated_four/tools/confirm_launch.py --wait 1800

#!/usr/bin/env bash
# Actual unified diffs for review: rareddie_pairout_cellout against the
# rareddie_control files it copies/adapts (review/implementation.diff; model.py
# and data.py must show NO diff), plus the new files (review/new_files.diff).
set -uo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
C=rareddie_control
N=rareddie_pairout_cellout
{
  echo "### rareddie_control -> rareddie_pairout_cellout (model.py/data.py byte-identical: empty diffs)"
  cmp "$C/model.py" "$N/model.py" && echo "model.py: byte-identical"
  cmp "$C/data.py" "$N/data.py" && echo "data.py: byte-identical"
  diff -u "$C/__init__.py" "$N/__init__.py"
  diff -u "$C/train.py" "$N/train.py"
  diff -u "$C/configs/single_drugout.yaml" "$N/configs/pairout.yaml"
  diff -u "$C/configs/double_drugout.yaml" "$N/configs/cellout.yaml"
  diff -u "$C/env.sh" "$N/env.sh"
  diff -u "$C/status.py" "$N/status.py"
  diff -u "$C/jobs.py" "$N/jobs.py"
  diff -u "$C/launch.sh" "$N/launch.sh"
  diff -u "$C/confirm_launch.py" "$N/confirm_launch.py"
  diff -u "$C/monitor/monitor_loop.sh" "$N/monitor/monitor_loop.sh"
  diff -u "$C/monitor/start_monitor.sh" "$N/monitor/start_monitor.sh"
  diff -u "$C/monitor/drain_recovery_requests.py" "$N/monitor/drain_recovery_requests.py"
  diff -u "$C/monitor/prompt.md" "$N/monitor/prompt.md"
  diff -u "$C/tools/make_review.sh" "$N/tools/make_review.sh"
} > "$N/review/implementation.diff"
{
  diff -uN /dev/null "$N/tools/cohort_evidence.py"
} > "$N/review/new_files.diff"
wc -l "$N/review/implementation.diff" "$N/review/new_files.diff"

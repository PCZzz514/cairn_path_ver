#!/usr/bin/env bash
# Actual unified diffs for review: rareddie_control against the reused
# accelerated_four files it adapts (review/implementation.diff), plus the new
# files (review/new_files.diff). The original RareDDIE block is included for
# the CSE order comparison.
set -uo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
A=accelerated_four
C=rareddie_control
{
  echo "### accelerated_four no-path comparator -> rareddie_control"
  diff -u "$A/dsn/model.py" "$C/model.py"
  diff -u "$A/dsn/data.py" "$C/data.py"
  diff -u "$A/dsn/train.py" "$C/train.py"
  diff -u "$A/configs/no_path_single_drugout.yaml" "$C/configs/single_drugout.yaml"
  diff -u "$A/configs/no_path_double_drugout.yaml" "$C/configs/double_drugout.yaml"
  diff -u "$A/env.sh" "$C/env.sh"
  diff -u "$A/status.py" "$C/status.py"
  diff -u "$A/jobs.py" "$C/jobs.py"
  diff -u "$A/launch_four.sh" "$C/launch.sh"
  diff -u "$A/tools/confirm_launch.py" "$C/confirm_launch.py"
  diff -u "$A/monitor/monitor_loop.sh" "$C/monitor/monitor_loop.sh"
  diff -u "$A/monitor/start_monitor.sh" "$C/monitor/start_monitor.sh"
  diff -u "$A/monitor/drain_recovery_requests.py" "$C/monitor/drain_recovery_requests.py"
  diff -u "$A/monitor/prompt.md" "$C/monitor/prompt.md"
  echo "### original RareDDIE MVN_DDI_Block (references/RareDDIE/models_t_struc.py lines 79-145) for CSE order"
  sed -n '79,145p' references/RareDDIE/models_t_struc.py
} > "$C/review/implementation.diff"
{
  for f in "$C/__init__.py" "$C/tools/cse_equivalence.py" "$C/tools/make_review.sh"; do
    diff -uN /dev/null "$f"
  done
} > "$C/review/new_files.diff"
wc -l "$C/review/implementation.diff" "$C/review/new_files.diff"

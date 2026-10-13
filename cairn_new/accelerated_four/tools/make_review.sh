#!/usr/bin/env bash
# Actual unified diffs for review: accelerated_four copies against the original
# production files and the GPU3-reviewed model/data (review/implementation.diff),
# and all new files (review/new_files.diff).
set -uo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
A=accelerated_four
G=benchmarks/gpu3_speed/dsn_gpu3
{
  echo "### GPU3-reviewed model/data -> accelerated copy (expected: no differences)"
  diff -u "$G/model.py" "$A/dsn/model.py"
  diff -u "$G/data.py" "$A/dsn/data.py"
  echo "### original production -> accelerated copy"
  for f in __init__.py metrics.py model.py data.py train.py; do
    diff -u "cairn_dsn/$f" "$A/dsn/$f"
  done
  for arm in path_single_drugout path_double_drugout no_path_single_drugout no_path_double_drugout; do
    diff -u "configs/$arm.yaml" "$A/configs/$arm.yaml"
  done
  diff -u scripts/server_jobs.py "$A/jobs.py"
  diff -u scripts/monitor_loop.sh "$A/monitor/monitor_loop.sh"
  diff -u scripts/start_monitor.sh "$A/monitor/start_monitor.sh"
  diff -u scripts/drain_recovery_requests.py "$A/monitor/drain_recovery_requests.py"
  diff -u docs/server_monitor_prompt.md "$A/monitor/prompt.md"
  diff -u scripts/launch_four.sh "$A/launch_four.sh"
} > "$A/review/implementation.diff"
{
  for f in "$A/__init__.py" "$A/env.sh" "$A/status.py" "$A/monitor/notify.py" "$A"/tools/*.py "$A"/tools/*.sh; do
    diff -uN /dev/null "$f"
  done
} > "$A/review/new_files.diff"
wc -l "$A/review/implementation.diff" "$A/review/new_files.diff"

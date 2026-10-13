#!/usr/bin/env bash
# Actual unified diffs for review: every pairout_cellout file against the
# reviewed file it copies/adapts (review/implementation.diff; empty sections
# mean byte-identical copies), plus genuinely new files (review/new_files.diff).
set -uo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
A=accelerated_four
R=rareddie_control
P=pairout_cellout
pair() {
  echo "### $1 -> $2"
  diff -u "$1" "$2"
}
{
  echo "### Unchanged copies (no diff output expected)"
  pair "$A/dsn/model.py" "$P/dsn/model.py"
  pair "$A/dsn/data.py" "$P/dsn/data.py"
  pair "$A/dsn/metrics.py" "$P/dsn/metrics.py"
  pair "$A/dsn/__init__.py" "$P/dsn/__init__.py"
  pair cairn_dsn/paths.py "$P/paths.py"
  echo "### Preparation/audit adapted from the original cairn_dsn"
  pair cairn_dsn/prepare.py "$P/prepare.py"
  pair cairn_dsn/audit.py "$P/audit.py"
  echo "### Trainer and configs adapted from the reviewed accelerated_four path arm"
  pair "$A/dsn/train.py" "$P/dsn/train.py"
  pair "$A/configs/path_single_drugout.yaml" "$P/configs/pairout.yaml"
  pair "$A/configs/path_single_drugout.yaml" "$P/configs/cellout.yaml"
  echo "### Two-arm controller/monitor adapted from rareddie_control"
  pair "$R/env.sh" "$P/env.sh"
  pair "$R/status.py" "$P/status.py"
  pair "$R/jobs.py" "$P/jobs.py"
  pair "$R/launch.sh" "$P/launch.sh"
  pair "$R/confirm_launch.py" "$P/confirm_launch.py"
  pair "$R/monitor/monitor_loop.sh" "$P/monitor/monitor_loop.sh"
  pair "$R/monitor/start_monitor.sh" "$P/monitor/start_monitor.sh"
  pair "$R/monitor/drain_recovery_requests.py" "$P/monitor/drain_recovery_requests.py"
  pair "$R/monitor/prompt.md" "$P/monitor/prompt.md"
  pair "$R/tools/make_review.sh" "$P/tools/make_review.sh"
} > "$P/review/implementation.diff"
{
  diff -uN /dev/null "$P/__init__.py"
} > "$P/review/new_files.diff"
wc -l "$P/review/implementation.diff" "$P/review/new_files.diff"

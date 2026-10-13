#!/usr/bin/env bash
# Actual unified diffs for review: dsn_pairout_cellout against the
# rareddie_pairout_cellout files it adapts (review/implementation.diff), cmp
# byte-equality of dsn/model.py and dsn/data.py with accelerated_four/dsn (must
# show NO diff), plus the new files (review/new_files.diff).
set -uo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
R=rareddie_pairout_cellout
A=accelerated_four/dsn
N=dsn_pairout_cellout
{
  echo "### accelerated_four/dsn -> dsn_pairout_cellout/dsn (byte-identical: cmp, empty diffs)"
  cmp "$A/model.py" "$N/dsn/model.py" && echo "dsn/model.py: byte-identical to $A/model.py"
  cmp "$A/data.py" "$N/dsn/data.py" && echo "dsn/data.py: byte-identical to $A/data.py"
  diff -u "$A/model.py" "$N/dsn/model.py"
  diff -u "$A/data.py" "$N/dsn/data.py"
  echo "### rareddie_pairout_cellout -> dsn_pairout_cellout (adapted)"
  diff -u "$R/__init__.py" "$N/__init__.py"
  diff -u "$R/train.py" "$N/train.py"
  diff -u "$R/configs/pairout.yaml" "$N/configs/pairout.yaml"
  diff -u "$R/configs/cellout.yaml" "$N/configs/cellout.yaml"
  diff -u "$R/env.sh" "$N/env.sh"
  diff -u "$R/status.py" "$N/status.py"
  diff -u "$R/jobs.py" "$N/jobs.py"
  diff -u "$R/launch.sh" "$N/launch.sh"
  diff -u "$R/confirm_launch.py" "$N/confirm_launch.py"
  diff -u "$R/monitor/monitor_loop.sh" "$N/monitor/monitor_loop.sh"
  diff -u "$R/monitor/start_monitor.sh" "$N/monitor/start_monitor.sh"
  diff -u "$R/monitor/drain_recovery_requests.py" "$N/monitor/drain_recovery_requests.py"
  diff -u "$R/monitor/prompt.md" "$N/monitor/prompt.md"
  diff -u "$R/tools/cohort_evidence.py" "$N/tools/cohort_evidence.py"
  diff -u "$R/tools/make_review.sh" "$N/tools/make_review.sh"
  echo "### $R/data.py (molecule-only Collator) is replaced by the unchanged $A/data.py Collator(with_path=False)"
  echo "### replaced by dsn/model.py (unchanged DSNRareDDI) + inventory.py: $R/model.py GROUPS/parameter_inventory"
  diff -u <(sed -n '/^GROUPS = /,$p' "$R/model.py") "$N/inventory.py"
} > "$N/review/implementation.diff"
{
  diff -uN /dev/null "$N/dsn/__init__.py"
  diff -uN /dev/null "$N/inventory.py"
} > "$N/review/new_files.diff"
wc -l "$N/review/implementation.diff" "$N/review/new_files.diff"

#!/usr/bin/env bash
# Write the actual unified diffs of the isolated copy against the originals
# (review/implementation.diff) and of all new benchmark files (review/new_files.diff).
set -uo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
B=benchmarks/gpu3_speed
{
  for f in __init__.py model.py data.py train.py metrics.py; do
    diff -u "cairn_dsn/$f" "$B/dsn_gpu3/$f"
  done
  diff -u configs/path_single_drugout.yaml "$B/configs/path_single_drugout.yaml"
} > "$B/review/implementation.diff"
{
  for f in "$B/dsn_gpu3/status.py" "$B/env.sh" "$B/run_benchmark.sh" "$B"/tools/*.py "$B/tools/make_review.sh" \
           logs/gpu3_speed/run_evidence.sh logs/gpu3_speed/launch_benchmark.sh; do
    diff -uN /dev/null "$f"
  done
} > "$B/review/new_files.diff"
wc -l "$B/review/implementation.diff" "$B/review/new_files.diff"

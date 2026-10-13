#!/usr/bin/env bash
# FOREGROUND GPU3 evidence (numerical OLD vs NEW + measured timing). No training.
# Usage: bash logs/gpu3_speed/run_evidence.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/../../benchmarks/gpu3_speed/env.sh"
LOG="$LOG_DIR/evidence_run.log"
echo "=== evidence start $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES GPU3_UUID=$GPU3_UUID" | tee -a "$LOG"
"$PYTHON_BIN" "$BENCH_DIR/tools/evidence.py" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
echo "=== evidence exit=$status $(date -Is)" | tee -a "$LOG"
exit "$status"

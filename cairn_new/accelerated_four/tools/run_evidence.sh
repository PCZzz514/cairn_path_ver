#!/usr/bin/env bash
# FOREGROUND evidence for all four arms (no training): path arms on CPU only,
# no-path single on physical GPU6, no-path double on physical GPU7.
# Usage: bash accelerated_four/tools/run_evidence.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
LOG="$LOG_DIR/evidence_run.log"
echo "=== evidence start $(date -Is) host=$(hostname)" | tee -a "$LOG"
declare -A DEVICE=([path_single_drugout]='' [path_double_drugout]='' [no_path_single_drugout]=6 [no_path_double_drugout]=7)
pids=()
for arm in path_single_drugout path_double_drugout no_path_single_drugout no_path_double_drugout; do
  CUDA_VISIBLE_DEVICES="${DEVICE[$arm]}" "$PYTHON_BIN" "$ACCEL_DIR/tools/evidence.py" --arm "$arm" \
    > "$LOG_DIR/evidence_$arm.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
for arm in path_single_drugout path_double_drugout no_path_single_drugout no_path_double_drugout; do
  tail -n 1 "$LOG_DIR/evidence_$arm.log" | tee -a "$LOG"
done
"$PYTHON_BIN" "$ACCEL_DIR/tools/summarize_evidence.py" 2>&1 | tee -a "$LOG" || status=1
echo "=== evidence exit=$status $(date -Is)" | tee -a "$LOG"
exit "$status"

#!/usr/bin/env bash
# Detached body started ONLY by logs/gpu3_speed/launch_benchmark.sh: exactly the
# five-epoch GPU3 run, then the completion summarizer (complete or failed).
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
set +e
echo "$$" > "$LOG_DIR/benchmark_runner.pid"
echo "=== runner pid=$$ ppid=$PPID sid=$(ps -o sid= -p $$ | tr -d ' ') start $(date -Is) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES GPU3_UUID=$GPU3_UUID"
child=
trap 'echo "=== runner received signal $(date -Is)"; [ -n "$child" ] && kill -TERM "$child"' TERM INT HUP
"$PYTHON_BIN" -m dsn_gpu3.train --config "$BENCH_DIR/configs/path_single_drugout.yaml" &
child=$!
echo "$child" > "$LOG_DIR/benchmark_train.pid"
"$PYTHON_BIN" -c "from dsn_gpu3.status import update_status; update_status('training', run={'runner_pid': $$, 'train_pid': $child, 'log': '$LOG_DIR/benchmark.log', 'started_at': '$(date -Is)'})"
wait "$child"
rc=$?
# A trapped signal interrupts wait; collect the real training exit status.
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
echo "=== training exit=$rc $(date -Is)"
"$PYTHON_BIN" "$BENCH_DIR/tools/summarize.py" --exit-code "$rc"
echo "=== runner done summarizer_exit=$? $(date -Is)"

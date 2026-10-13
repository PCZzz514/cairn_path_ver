#!/usr/bin/env bash
# Direct host launch of the five-epoch GPU3 benchmark. Codex runs this ONLY
# after writing logs/gpu3_speed/review_approval.json. Starts an independent
# detached session (setsid, orphaned -> PPID 1), records PID, waits for the
# training start line, and returns. Usage: bash logs/gpu3_speed/launch_benchmark.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/../../benchmarks/gpu3_speed/env.sh"
"$PYTHON_BIN" "$BENCH_DIR/tools/launch_gate.py"
LOGFILE="$LOG_DIR/benchmark.log"
rm -f "$LOG_DIR/benchmark_runner.pid" "$LOG_DIR/benchmark_train.pid"
echo "=== launch $(date -Is) by $(id -un) from pid $$" >> "$LOGFILE"
# Subshell exits immediately, so the setsid session is reparented away from us.
( setsid nohup bash "$BENCH_DIR/run_benchmark.sh" >> "$LOGFILE" 2>&1 < /dev/null & echo "$!" > "$LOG_DIR/benchmark_launch_bang.pid" )
for _ in $(seq 60); do [ -s "$LOG_DIR/benchmark_runner.pid" ] && break; sleep 1; done
PID="$(cat "$LOG_DIR/benchmark_runner.pid")"
sleep 2
read -r RPPID PGID SID <<< "$(ps -o ppid=,pgid=,sid= -p "$PID" || echo -1 -1 -1)"
# Wait (bounded) for real training progress: the start line after data/model load.
for _ in $(seq 300); do grep -q '"phase": "start"' "$LOGFILE" && break; kill -0 "$PID" 2>/dev/null || break; sleep 1; done
STARTED=$(grep -c '"phase": "start"' "$LOGFILE" || true)
"$PYTHON_BIN" - "$PID" "$RPPID" "$PGID" "$SID" "$LOGFILE" "$STARTED" "$(cat "$LOG_DIR/benchmark_launch_bang.pid")" <<'PY'
import json, sys, time
from pathlib import Path
from dsn_gpu3.status import update_status
pid, ppid, pgid, sid, log, started, bang = sys.argv[1:]
train = Path(log).with_name('benchmark_train.pid')
launch = {'runner_pid': int(pid), 'runner_ppid': int(ppid), 'ppid_is_1': ppid == '1', 'pgid': int(pgid), 'sid': int(sid),
          'setsid_pid_from_launcher': int(bang), 'train_pid': int(train.read_text()) if train.exists() else None,
          'runner_alive': Path(f'/proc/{pid}').exists(), 'training_start_line_seen': started != '0',
          'log': log, 'launched_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
          'expected_outputs': ['benchmarks/gpu3_speed/outputs/path_single_drugout/history.jsonl',
                               'logs/gpu3_speed/five_epoch_summary.json', 'logs/gpu3_speed/five_epoch_summary.md']}
Path(log).with_name('launch.json').write_text(json.dumps(launch, indent=2) + '\n')
update_status('training' if launch['runner_alive'] else 'launch_failed', launch=launch)
print(json.dumps(launch, indent=2))
if not launch['ppid_is_1']:
    print(f'WARNING: runner PPID is {ppid}, not 1 (a subreaper adopted it); verify with ps before relying on it.', file=sys.stderr)
PY
ps -o pid,ppid,pgid,sid,lstart,args -p "$PID,$(cat "$LOG_DIR/benchmark_train.pid" 2>/dev/null || echo "$PID")" || true
tail -n 5 "$LOGFILE"

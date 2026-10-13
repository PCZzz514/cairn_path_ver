#!/usr/bin/env bash
# Detached pairout_cellout monitor daemon (started only by start_monitor.sh).
# Pattern of accelerated_four/monitor/monitor_loop.sh (left untouched):
# deterministic status, one gpt-6.1-sol LOW Codex check, host drain of recovery
# requests, after status; first check immediately, then one every 2400 seconds.
# Local state only (logs/pairout_cellout/monitor, status.json); no notifications.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
MONITOR_DIR="$LOG_DIR/monitor"
mkdir -p "$MONITOR_DIR"
exec 9>"$MONITOR_DIR/daemon.lock"
flock -n 9 || exit 0
echo "$$" > "$MONITOR_DIR/monitor.pid"
export CODEX_HOME="$ROOT_DIR/.codex_runtime"
# The monitor never needs a GPU; training jobs run in their own sessions.
export CUDA_VISIBLE_DEVICES=''
# Use the existing proxy service without shell initialization/history writes.
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890 no_proxy=localhost,127.0.0.1,::1
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export ALL_PROXY="$all_proxy" NO_PROXY="$no_proxy"
cd "$ROOT_DIR"
CLI_FAILURES=0
trap 'date -u +%FT%TZ > "$MONITOR_DIR/stopped_at.txt"' EXIT
status_to() {
  # A status/verifier error is recorded and never stops monitoring.
  "$PYTHON_BIN" pairout_cellout/jobs.py --status > "$1" 2> "$1.err" \
    || printf '{"all_complete": false, "status_error": true, "runs": {}}\n' > "$1"
}
record_status() {
  # monitor section of logs/pairout_cellout/status.json; failures never stop monitoring.
  "$PYTHON_BIN" - "$MONITOR_DIR" "$CHECK_ID" "$1" "${CODEX_EXIT:-}" <<'PY' || true
import json, sys
from pathlib import Path
from pairout_cellout.status import update_status
monitor, check, outcome, codex_exit = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
after = monitor / 'checks' / check / 'after.json'
runs = json.loads(after.read_text()).get('runs', {}) if after.exists() else {}
update_status('monitor_' + outcome, section='monitor', pid=int((monitor / 'monitor.pid').read_text()),
              last_check=check, last_codex_exit=int(codex_exit) if codex_exit else None,
              last_successful_check=(monitor / 'last_successful_check.txt').read_text().strip()
              if (monitor / 'last_successful_check.txt').exists() else None,
              next_check=(monitor / 'next_check.txt').read_text().strip() if (monitor / 'next_check.txt').exists() else None,
              interval_seconds=2400, model='gpt-6.1-sol low', outcome=outcome,
              runs={arm: {k: r.get(k) for k in ['state', 'attempt', 'recoveries', 'history_last_epoch',
                                                 'history_best_validation_macro_f1_85', 'history_bad_epochs']}
                    for arm, r in runs.items()})
PY
}
while true; do
  CHECK_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
  CHECK_DIR="$MONITOR_DIR/checks/$CHECK_ID"
  CODEX_EXIT=''
  mkdir -p "$CHECK_DIR"
  date -u +%FT%TZ > "$MONITOR_DIR/last_check.txt"
  date -u -d '+2400 seconds' +%FT%TZ > "$MONITOR_DIR/next_check.txt"
  # Check lock covers both state reads, Codex, and the host recovery drain.
  exec 7>"$MONITOR_DIR/check.lock"
  flock 7
  status_to "$CHECK_DIR/before.json"
  cp "$CHECK_DIR/before.json" "$MONITOR_DIR/snapshot_summary.json"
  if "$PYTHON_BIN" - "$CHECK_DIR/before.json" <<'PY'
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))['all_complete'] else 1)
PY
  then
    cp "$CHECK_DIR/before.json" "$CHECK_DIR/after.json"
    cp "$CHECK_DIR/before.json" "$MONITOR_DIR/final_summary.json"
    echo 'complete' > "$MONITOR_DIR/state.txt"
    rm -f "$MONITOR_DIR/next_check.txt"
    record_status complete
    echo 'Both selected-best checkpoint tests and complete exports verified.'
    exit 0
  fi
  echo 'monitoring' > "$MONITOR_DIR/state.txt"
  printf '%s model=gpt-6.1-sol reasoning=low interval_seconds=2400\n' "$CHECK_ID"
  # A CLI timeout affects ONLY this check, never a training process.
  if timeout --signal=TERM --kill-after=30s 1200s \
      /data/interns/zhangpc/.local/bin/codex exec \
      --dangerously-bypass-approvals-and-sandbox --ignore-user-config --ephemeral --skip-git-repo-check \
      -c 'model_reasoning_effort="low"' -m gpt-6.1-sol -C "$ROOT_DIR" \
      -o "$CHECK_DIR/codex_final.txt" - < pairout_cellout/monitor/prompt.md \
      > "$CHECK_DIR/codex.log" 2>&1; then
    CODEX_EXIT=0
    CLI_FAILURES=0
  else
    CODEX_EXIT=$?
    CLI_FAILURES=$((CLI_FAILURES + 1))
    printf '%s exit=%s consecutive_failures=%s\n' "$CHECK_ID" "$CODEX_EXIT" "$CLI_FAILURES" >> "$MONITOR_DIR/cli_failures.log"
  fi
  echo "$CODEX_EXIT" > "$CHECK_DIR/codex_exit.txt"
  # Launch recovery only here, outside the CLI timeout and its PID namespace.
  if "$PYTHON_BIN" pairout_cellout/monitor/drain_recovery_requests.py \
      > "$CHECK_DIR/recovery_drain.log" 2>&1; then
    :
  else
    DRAIN_EXIT=$?
    printf '%s drain_exit=%s\n' "$CHECK_ID" "$DRAIN_EXIT" >> "$MONITOR_DIR/recovery_drain_errors.log"
  fi
  status_to "$CHECK_DIR/after.json"
  cp "$CHECK_DIR/after.json" "$MONITOR_DIR/snapshot_summary.json"
  # Successful check: Codex exit 0 and after.json persisted (next_check.txt above).
  [[ "$CODEX_EXIT" == 0 ]] && echo "$CHECK_ID" > "$MONITOR_DIR/last_successful_check.txt"
  OUTCOME="$("$PYTHON_BIN" - "$CHECK_DIR/after.json" <<'PY' || echo monitoring
import json, sys
state = json.load(open(sys.argv[1]))
if state['all_complete']:
    print('complete')
elif state['runs'] and all(r['state'] in ('complete', 'exhausted') for r in state['runs'].values()):
    print('blocked_recovery_budget')
else:
    print('monitoring')
PY
  )"
  echo "$OUTCOME" > "$MONITOR_DIR/state.txt"
  if [[ "$OUTCOME" != monitoring ]]; then
    cp "$CHECK_DIR/after.json" "$MONITOR_DIR/final_summary.json"
    rm -f "$MONITOR_DIR/next_check.txt"
    record_status "$OUTCOME"
    [[ "$OUTCOME" == complete ]] && exit 0
    echo "Monitor blocked: $OUTCOME; see final_summary.json and attempt logs" >&2
    exit 1
  fi
  record_status monitoring
  flock -u 7
  # Maintain a 2400-second interval between check starts, including CLI time.
  NEXT_EPOCH="$(date -u -d "$(cat "$MONITOR_DIR/next_check.txt")" +%s)"
  NOW_EPOCH="$(date -u +%s)"
  if (( NEXT_EPOCH > NOW_EPOCH )); then
    sleep "$((NEXT_EPOCH - NOW_EPOCH))"
  fi
done

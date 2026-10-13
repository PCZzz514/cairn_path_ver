#!/usr/bin/env bash
# Detached crt_kpaths monitor daemon (started by start_monitor.sh). Every 2400 s:
# deterministic status, keep the dispatcher alive, one gpt-6.1-sol LOW Codex
# check (CPU only), host drain of Codex recovery requests, status again.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
MONITOR_DIR="$LOG_DIR/monitor"
mkdir -p "$MONITOR_DIR"
exec 9>"$MONITOR_DIR/daemon.lock"
flock -n 9 || exit 0
echo "$$" > "$MONITOR_DIR/monitor.pid"
export CODEX_HOME="$ROOT_DIR/.codex_runtime"
export CUDA_VISIBLE_DEVICES=''
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890 no_proxy=localhost,127.0.0.1,::1
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy" NO_PROXY="$no_proxy"
trap 'date -u +%FT%TZ > "$MONITOR_DIR/stopped_at.txt"' EXIT
all_complete() {
  "$PYTHON_BIN" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("all_complete") else 1)' "$1"
}
status_to() {
  "$PYTHON_BIN" crt_kpaths/jobs.py --status > "$1" 2> "$1.err" \
    || printf '{"all_complete": false, "status_error": true, "runs": {}}\n' > "$1"
}
while true; do
  CHECK_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
  CHECK_DIR="$MONITOR_DIR/checks/$CHECK_ID"
  mkdir -p "$CHECK_DIR"
  date -u +%FT%TZ > "$MONITOR_DIR/last_check.txt"
  date -u -d '+2400 seconds' +%FT%TZ > "$MONITOR_DIR/next_check.txt"
  status_to "$CHECK_DIR/before.json"
  if all_complete "$CHECK_DIR/before.json"; then
    cp "$CHECK_DIR/before.json" "$MONITOR_DIR/final_summary.json"
    echo complete > "$MONITOR_DIR/state.txt"
    rm -f "$MONITOR_DIR/next_check.txt"
    exit 0
  fi
  echo monitoring > "$MONITOR_DIR/state.txt"
  # Idempotent: returns the live dispatcher or restarts a dead one (it only
  # starts never-launched jobs whose dependencies are complete).
  "$PYTHON_BIN" crt_kpaths/jobs.py --launch > "$CHECK_DIR/dispatcher.json" 2>&1 || true
  if timeout --signal=TERM --kill-after=30s 1200s \
      /data/interns/zhangpc/.local/bin/codex exec \
      --skip-git-repo-check --ephemeral --ignore-user-config \
      --dangerously-bypass-approvals-and-sandbox \
      -c 'model_reasoning_effort="low"' -m gpt-6.1-sol -C "$ROOT_DIR" \
      -o "$CHECK_DIR/codex_final.txt" - < crt_kpaths/monitor/prompt.md \
      > "$CHECK_DIR/codex.log" 2>&1; then
    echo 0 > "$CHECK_DIR/codex_exit.txt"
  else
    echo "$?" > "$CHECK_DIR/codex_exit.txt"
    printf '%s exit=%s\n' "$CHECK_ID" "$(cat "$CHECK_DIR/codex_exit.txt")" >> "$MONITOR_DIR/cli_failures.log"
  fi
  # Recoveries are launched here, outside the Codex sandbox/timeout.
  "$PYTHON_BIN" crt_kpaths/jobs.py --drain-requests > "$CHECK_DIR/recovery_drain.json" 2>&1 || true
  status_to "$CHECK_DIR/after.json"
  cp "$CHECK_DIR/after.json" "$MONITOR_DIR/snapshot_summary.json"
  if all_complete "$CHECK_DIR/after.json"; then
    cp "$CHECK_DIR/after.json" "$MONITOR_DIR/final_summary.json"
    echo complete > "$MONITOR_DIR/state.txt"
    rm -f "$MONITOR_DIR/next_check.txt"
    exit 0
  fi
  NEXT_EPOCH="$(date -u -d "$(cat "$MONITOR_DIR/next_check.txt")" +%s)"
  NOW_EPOCH="$(date -u +%s)"
  if (( NEXT_EPOCH > NOW_EPOCH )); then sleep "$((NEXT_EPOCH - NOW_EPOCH))"; fi
done

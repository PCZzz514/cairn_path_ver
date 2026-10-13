#!/usr/bin/env bash
# Start the detached rareddie_pairout_cellout monitor daemon ONLY after both jobs have
# records and confirmed real progress. The daemon is double-forked (setsid -f,
# PPID 1, own session). Waits for its FIRST SUCCESSFUL check: Codex exit 0,
# after.json and next_check.txt persisted. Safe to repeat while it is live.
# Usage: bash rareddie_pairout_cellout/monitor/start_monitor.sh
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
export CUDA_VISIBLE_DEVICES=''
MONITOR_DIR="$LOG_DIR/monitor"
mkdir -p "$MONITOR_DIR"
# Serialize monitor launches separately from its long-lived daemon lock.
exec 8>"$MONITOR_DIR/launcher.lock"
flock -n 8 || { echo 'Another monitor launch is in progress' >&2; exit 1; }
if ! flock -n "$MONITOR_DIR/daemon.lock" true; then
  echo "Monitor is already running; see $MONITOR_DIR/monitor.pid"
  exit 0
fi
"$PYTHON_BIN" - "$MONITOR_DIR" <<'PY'
import subprocess
import sys
import time
from pathlib import Path
from rareddie_pairout_cellout.jobs import ARMS, STATE, process_start
from rareddie_pairout_cellout.confirm_launch import arm_state, ps
from rareddie_pairout_cellout.status import update_status
monitor = Path(sys.argv[1])
for arm in ARMS:
    if not (STATE / arm / 'record.json').is_file():
        raise SystemExit('Launch and confirm both jobs before starting the monitor')
    state = arm_state(arm)
    if not (state['progress_confirmed'] or state['exit'] is not None):
        raise SystemExit(f'{arm}: no confirmed real progress yet (fresh start, first update, 100-update line)')
if not Path('/data/interns/zhangpc/.local/bin/codex').is_file():
    raise SystemExit('Existing Codex CLI is unavailable')
root = Path.cwd()
(monitor / 'monitor.pid').unlink(missing_ok=True)
started = time.time()
with (monitor / 'daemon.log').open('a') as log:
    subprocess.run(['setsid', '-f', 'bash', str(root / 'rareddie_pairout_cellout/monitor/monitor_loop.sh')], cwd=root,
                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True, check=True)
for _ in range(300):
    if (monitor / 'monitor.pid').is_file() and (monitor / 'monitor.pid').read_text().strip():
        break
    time.sleep(0.1)
else:
    raise SystemExit(f'Monitor did not start; see {monitor}/daemon.log')
pid = int((monitor / 'monitor.pid').read_text())
# First check is immediate; Codex is bounded by 1200 s plus 30 s kill grace.
deadline = started + 1500
while time.time() < deadline:
    if process_start(pid) is None:
        raise SystemExit(f'Monitor PID {pid} exited before its first check completed; see {monitor}/daemon.log')
    checks = sorted((monitor / 'checks').glob(f'*_{pid}'))
    if checks and (checks[0] / 'after.json').is_file() and (checks[0] / 'codex_exit.txt').is_file():
        break
    time.sleep(5)
else:
    raise SystemExit(f'First monitor check not finished within 1500 s; daemon PID {pid} still running')
check = checks[0]
codex_exit = int((check / 'codex_exit.txt').read_text())
next_check = (monitor / 'next_check.txt').read_text().strip() if (monitor / 'next_check.txt').is_file() else None
success = codex_exit == 0 and next_check is not None
daemon_ps = ps(pid)
fields = {'pid': pid, 'ps': daemon_ps, 'ppid_is_1': (daemon_ps or '').split()[1:2] == ['1'],
          'state_dir': str(monitor), 'first_check': check.name, 'first_check_codex_exit': codex_exit,
          'first_check_after_json': str(check / 'after.json'), 'next_check': next_check,
          'interval_seconds': 2400, 'model': 'gpt-6.1-sol low'}
update_status('monitor_first_check_succeeded' if success else 'monitor_first_check_failed', section='monitor_start',
              **fields)
print(fields)
if not success:
    raise SystemExit(f'First monitor check NOT successful (codex exit {codex_exit}); daemon keeps its 2400 s schedule')
print(f'Detached rareddie_pairout_cellout monitor PID {pid}; first check succeeded; next check {next_check}')
PY

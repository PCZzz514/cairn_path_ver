#!/usr/bin/env bash
# Start the detached accelerated_four monitor daemon ONLY after all four jobs
# have records and confirmed real progress. Safe to repeat while it is live.
# Usage: bash accelerated_four/monitor/start_monitor.sh
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
from accelerated_four.jobs import ARMS, STATE
from accelerated_four.tools.confirm_launch import arm_state
from accelerated_four.status import update_status
monitor = Path(sys.argv[1])
for arm in ARMS:
    if not (STATE / arm / 'record.json').is_file():
        raise SystemExit('Launch and confirm all four jobs before starting the monitor')
    state = arm_state(arm)
    if not (state['progress_confirmed'] or state['exit'] is not None):
        raise SystemExit(f'{arm}: no confirmed real progress yet (start line, first step and 100-update line)')
if not Path('/data/interns/zhangpc/.local/bin/codex').is_file():
    raise SystemExit('Existing Codex CLI is unavailable')
root = Path.cwd()
with (monitor / 'daemon.log').open('a') as log:
    process = subprocess.Popen(['bash', str(root / 'accelerated_four/monitor/monitor_loop.sh')],
                               cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, close_fds=True)
(monitor / 'monitor.pid').write_text(str(process.pid) + '\n')
# Confirm initialization before reporting a successful detached launch.
for _ in range(600):
    if process.poll() is not None:
        raise SystemExit(f'Monitor exited during startup; see {monitor}/daemon.log')
    if (monitor / 'next_check.txt').is_file() and any((monitor / 'checks').glob(f'*_{process.pid}/before.json')):
        break
    time.sleep(0.1)
else:
    raise SystemExit('Monitor startup not confirmed; inspect daemon.log and daemon.lock before retrying')
update_status('monitor_started', monitor={'pid': process.pid, 'state_dir': str(monitor),
                                          'next_check': (monitor / 'next_check.txt').read_text().strip(),
                                          'interval_seconds': 2400, 'model': 'gpt-6.1-sol low'})
print(f'Detached accelerated_four monitor PID {process.pid}; checks every 2400 seconds')
PY

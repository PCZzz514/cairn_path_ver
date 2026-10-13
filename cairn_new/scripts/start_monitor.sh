#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
export CUDA_VISIBLE_DEVICES=''
mkdir -p logs/server_monitor
# Serialize monitor launches separately from its long-lived daemon lock.
exec 8>logs/server_monitor/launcher.lock
flock -n 8 || { echo 'Another monitor launch is in progress' >&2; exit 1; }
if ! flock -n logs/server_monitor/daemon.lock true; then
  echo 'Monitor is already running; see logs/server_monitor/monitor.pid'
  exit 0
fi
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import subprocess
import time
from scripts.server_jobs import ARMS, STATE
for arm in ARMS:
    if not (STATE / arm / 'record.json').is_file():
        raise SystemExit('Launch and confirm all four jobs before starting the monitor')
if not Path('/data/interns/zhangpc/.local/bin/codex').is_file():
    raise SystemExit('Existing Codex CLI is unavailable')
root = Path.cwd()
with (root / 'logs/server_monitor/daemon.log').open('a') as log:
    process = subprocess.Popen(['bash', str(root / 'scripts/monitor_loop.sh')],
                               cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, close_fds=True)
(root / 'logs/server_monitor/monitor.pid').write_text(str(process.pid) + '\n')
# Confirm initialization before reporting a successful detached launch.
for _ in range(50):
    if process.poll() is not None:
        raise SystemExit('Monitor exited during startup; see logs/server_monitor/daemon.log')
    checks = root / 'logs/server_monitor'
    if (checks / 'next_check.txt').is_file() and any(checks.glob(f'*_{process.pid}/before.json')):
        break
    time.sleep(0.1)
else:
    raise SystemExit('Monitor startup not confirmed; inspect daemon.log and daemon.lock before retrying')
print(f'Detached server monitor PID {process.pid}; checks every 2400 seconds')
PY

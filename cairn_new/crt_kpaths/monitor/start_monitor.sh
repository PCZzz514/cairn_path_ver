#!/usr/bin/env bash
# Start the detached crt_kpaths monitor and confirm its FIRST check really
# finished: codex_exit.txt == 0, after.json written, next_check.txt present.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
export CUDA_VISIBLE_DEVICES=''
MONITOR_DIR="$LOG_DIR/monitor"
mkdir -p "$MONITOR_DIR"
if ! flock -n "$MONITOR_DIR/daemon.lock" true; then
  echo "Monitor already running: PID $(cat "$MONITOR_DIR/monitor.pid")"; exit 0
fi
"$PYTHON_BIN" - "$MONITOR_DIR" <<'PY'
import subprocess, sys, time
from pathlib import Path
monitor = Path(sys.argv[1])
with (monitor / 'daemon.log').open('a') as log:
    process = subprocess.Popen(['bash', 'crt_kpaths/monitor/monitor_loop.sh'], stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
deadline = time.time() + 1500
while time.time() < deadline:
    if process.poll() is not None:
        raise SystemExit(f'Monitor exited during its first check; see {monitor}/daemon.log')
    done = [d for d in (monitor / 'checks').glob(f'*_{process.pid}') if (d / 'after.json').exists()]
    if done:
        code = (done[0] / 'codex_exit.txt').read_text().strip()
        next_check = (monitor / 'next_check.txt').read_text().strip()
        print(f'Monitor PID {process.pid}; first check {done[0].name}: codex_exit={code}, after.json written, '
              f'next_check={next_check} (every 2400 s); Codex message {done[0]}/codex_final.txt')
        raise SystemExit(0 if code == '0' else 1)
    time.sleep(5)
raise SystemExit('First Codex check not finished within 1500 s; inspect daemon.log')
PY

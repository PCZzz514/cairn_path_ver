"""Consume monitor recovery requests in the host daemon, once per request."""
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
REQUESTS = ROOT / 'logs/server_monitor/recovery_requests'


def write_record(path, record):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    if not REQUESTS.exists():
        return
    processed = REQUESTS / 'processed'
    processed.mkdir(exist_ok=True)
    for pending in sorted(REQUESTS.glob('*.json')):
        # Claim before invoking the controller. An interrupted drain leaves
        # evidence here and never automatically retries this request.
        claimed = processed / f'{pending.stem}.{time.time_ns()}.json'
        pending.rename(claimed)
        record = {'request_file': pending.name, 'started_at': time.time(),
                  'result': 'processing', 'exit_code': None,
                  'stdout': '', 'stderr': ''}
        try:
            raw = claimed.read_text()
            record['raw_request'] = raw
            request = json.loads(raw)
            if (not isinstance(request, dict)
                    or not isinstance(request.get('arm'), str)
                    or not isinstance(request.get('reason'), str)
                    or not request['reason'].strip()
                    or type(request.get('reduce_batch')) is not bool):
                raise ValueError('Request requires arm, nonempty reason, and reduce_batch boolean')
            record['request'] = {key: request[key] for key in ('arm', 'reason', 'reduce_batch')}
            command = [sys.executable, str(ROOT / 'scripts/server_jobs.py'),
                       '--recover', request['arm'], '--reason', request['reason']]
            if request['reduce_batch']:
                command.append('--reduce-batch')
            record['command'] = command
            write_record(claimed, record)
            # Inherit the host daemon's interpreter/runtime. The existing
            # controller alone decides whether recovery is allowed.
            result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, errors='replace')
            record.update(result='succeeded' if result.returncode == 0 else 'rejected',
                          exit_code=result.returncode,
                          stdout=result.stdout, stderr=result.stderr)
        except Exception as exc:
            record.update(result='error', error=f'{type(exc).__name__}: {exc}')
        record['finished_at'] = time.time()
        write_record(claimed, record)


if __name__ == '__main__':
    main()

"""Detached crt_kpaths job controller: status, dependency dispatch, bounded resume.

Jobs (GPU visibility is exactly one device per job):
  step1_{single,double}_drugout        GPU4, concurrent (extraction + heads)
  step2_kpaths_{single,double}_drugout GPU5 / GPU6, concurrent
  step3_{single,double}_drugout        GPU7, serial; each after its own Step2
                                       test_complete and its Step1 lr selection
--dispatch is the session-independent daemon that starts each job once when its
dependencies are complete; it never restarts a failed job. --recover resumes a
failed job from its own last state (Step2: last.pt --resume or --test-only;
Step1/3: features/heads already written are reused), never from scratch, never
with a different batch size, at most MAX_RECOVERIES times.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CRT = ROOT / 'crt_kpaths'
STATE = ROOT / 'logs/crt_kpaths/jobs'
OUTPUTS = CRT / 'outputs'
REQUESTS = ROOT / 'logs/crt_kpaths/monitor/recovery_requests.jsonl'
MAX_RECOVERIES = 3
JOBS = {
    'step1_single_drugout': {'gpu': 4, 'module': 'crt_kpaths.crt', 'deps': []},
    'step1_double_drugout': {'gpu': 4, 'module': 'crt_kpaths.crt', 'deps': []},
    'step2_kpaths_single_drugout': {'gpu': 5, 'module': 'crt_kpaths.train', 'deps': []},
    'step2_kpaths_double_drugout': {'gpu': 6, 'module': 'crt_kpaths.train', 'deps': []},
    'step3_single_drugout': {'gpu': 7, 'module': 'crt_kpaths.crt',
                             'deps': ['step1_single_drugout', 'step2_kpaths_single_drugout']},
    'step3_double_drugout': {'gpu': 7, 'module': 'crt_kpaths.crt',
                             'deps': ['step1_double_drugout', 'step2_kpaths_double_drugout']},
}
SERIAL_GPU = 7


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


@contextmanager
def registry_lock():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def process_start(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def busy(job):
    # The training child inherits job.lock, so a live child counts even if
    # its wrapper died.
    with (STATE / job / 'job.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def live(record):
    return (record['process_start'] is not None and process_start(record['pid']) == record['process_start']) \
        or busy(record['job'])


def output_of(job):
    return OUTPUTS / job


def complete(job):
    output = output_of(job)
    if not (output / 'test_complete.json').is_file():
        return False
    if job.startswith('step2'):
        marker, metrics = read_json(output / 'test_complete.json'), read_json(output / 'test_metrics.json')
        return marker['selected_epoch'] == metrics['selected_epoch'] and metrics['selection'] == 'validation only'
    return (output / 'summary.json').is_file()


def history(output):
    path = output / 'history.jsonl'
    return [json.loads(line) for line in path.read_text().split('\n')[:-1]] if path.exists() else []


def last_event(path):
    try:
        with open(path, 'rb') as stream:
            stream.seek(max(0, stream.seek(0, 2) - 65536))
            lines = stream.read().decode(errors='replace').splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        if line.startswith('{'):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            return {k: event[k] for k in ['phase', 'epoch', 'update', 'split', 'step', 'arm', 'base_lr', 'epoch_lr', 'lr', 'seed',
                                          'val_macro_f1_85', 'best_validation_macro_f1_85', 'bad_epochs',
                                          'epoch_seconds', 'seconds'] if k in event}
    return None


def job_state(job):
    record_path = STATE / job / 'record.json'
    if not record_path.exists():
        waiting = [d for d in JOBS[job]['deps'] if not complete(d)]
        return {'state': 'waiting' if waiting else 'not_launched', 'waiting_for': waiting, 'gpu': JOBS[job]['gpu']}
    record = read_json(record_path)
    running = live(record)
    done = not running and complete(job)
    state = {'state': 'live' if running else 'complete' if done else
             'exhausted' if record['recoveries'] >= MAX_RECOVERIES else 'failed',
             'gpu': record['gpu'], 'pid': record['pid'], 'attempt': record['attempt'],
             'recoveries': record['recoveries'], 'mode': record['mode'], 'log': record['log'],
             'last_log_event': last_event(record['log']),
             'exit': read_json(Path(record['exit_file'])) if Path(record['exit_file']).exists() else None}
    if job.startswith('step2'):
        records = history(output_of(job))
        state.update(history_last_epoch=records[-1]['epoch'] if records else None,
                     best_validation_macro_f1_85=records[-1]['best_validation_macro_f1_85'] if records else None,
                     bad_epochs=records[-1]['bad_epochs'] if records else None)
    return state


def status():
    runs = {job: job_state(job) for job in JOBS}
    dispatcher = STATE / 'dispatcher.json'
    snapshot = {'checked_at': time.time(), 'runs': runs,
                'dispatcher': read_json(dispatcher) if dispatcher.exists() else None,
                'all_complete': all(r['state'] == 'complete' for r in runs.values())}
    if snapshot['dispatcher']:
        snapshot['dispatcher']['live'] = process_start(snapshot['dispatcher']['pid']) == snapshot['dispatcher']['process_start']
    write_json(STATE / 'status.json', snapshot)
    if snapshot['all_complete']:
        write_json(STATE / 'final_summary.json', snapshot)
    return snapshot


def command(job, mode):
    config = CRT / 'configs' / f'{job}.yaml'
    return [sys.executable, '-m', JOBS[job]['module'], '--config', str(config)] + ([mode] if mode else [])


def start(job, mode, recoveries=0, reason='dispatch: dependencies complete'):
    directory = STATE / job
    directory.mkdir(parents=True, exist_ok=True)
    old = read_json(directory / 'record.json') if (directory / 'record.json').exists() else None
    if old and live(old):
        raise RuntimeError(f'{job} is live')
    attempt = old['attempt'] + 1 if old else 1
    log = directory / f'attempt_{attempt}.log'
    record = {'job': job, 'gpu': JOBS[job]['gpu'], 'attempt': attempt, 'recoveries': recoveries,
              'started_at': time.time(), 'mode': mode or 'fresh', 'reason': reason, 'log': str(log),
              'exit_file': str(directory / f'attempt_{attempt}_exit.json'), 'command': command(job, mode)}
    launch_file = directory / f'attempt_{attempt}.json'
    write_json(launch_file, record)
    with log.open('x') as stream:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(launch_file)],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    record.update(pid=child.pid, process_start=process_start(child.pid))
    write_json(launch_file, record)
    write_json(directory / 'record.json', record)
    return record


def worker(launch_file):
    record = read_json(launch_file)
    with (STATE / record['job'] / 'job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(record['gpu']), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT))
        child = subprocess.Popen(record['command'], cwd=ROOT, env=env, pass_fds=(lock.fileno(),))
        write_json(launch_file.with_name(launch_file.stem + '_process.json'),
                   {'pid': child.pid, 'process_start': process_start(child.pid)})
        code = child.wait()
        write_json(Path(record['exit_file']), {'exit_code': code, 'finished_at': time.time()})
        return code if code >= 0 else 128 - code


def step2_resume_mode(job):
    """--resume from last.pt, or --test-only when its schedule is finished.

    The epoch after last.pt may already be in history (crash between append
    and publication); it is archived and dropped, then recomputed."""
    import torch
    output = output_of(job)
    if not (output / 'last.pt').exists():
        raise RuntimeError('No last.pt: refusing a silent restart from scratch; report to desktop')
    last = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
    cfg = last['config']
    records = history(output)
    kept = [r for r in records if r['epoch'] <= last['epoch']]
    if [r['epoch'] for r in kept] != list(range(1, last['epoch'] + 1)) or kept[-1] != last['metrics']:
        raise RuntimeError('history.jsonl incoherent with last.pt; not relaunched')
    if len(kept) != len(records):
        shutil.copyfile(output / 'history.jsonl', output / f'history.before_resume_{int(time.time())}.jsonl')
        temporary = output / 'history.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in kept))
        temporary.replace(output / 'history.jsonl')
    done = last['epoch'] >= cfg['epochs'] or last['bad_epochs'] >= cfg['patience']
    return '--test-only' if done else '--resume'


def recover(job, reason):
    record = read_json(STATE / job / 'record.json')
    if live(record) or complete(job):
        raise RuntimeError('Live or complete: no recovery')
    if record['recoveries'] >= MAX_RECOVERIES:
        raise RuntimeError('Recovery budget exhausted')
    if JOBS[job]['gpu'] == SERIAL_GPU and any(
            other != job and JOBS[other]['gpu'] == SERIAL_GPU and job_state(other)['state'] == 'live' for other in JOBS):
        raise RuntimeError('Another Step3 job is live on GPU7; Step3 is serial')
    mode = step2_resume_mode(job) if job.startswith('step2') else None
    return start(job, mode, record['recoveries'] + 1, reason)


def dispatch_once():
    started = []
    for job, spec in JOBS.items():
        if (STATE / job / 'record.json').exists() or not all(complete(d) for d in spec['deps']):
            continue
        if spec['gpu'] == SERIAL_GPU and any(
                JOBS[other]['gpu'] == SERIAL_GPU and (STATE / other / 'record.json').exists()
                and job_state(other)['state'] in ('live', 'failed') for other in JOBS):
            continue  # GPU7 runs Step3 serially; a failed one awaits recovery first
        started.append(start(job, None))
    return started


def dispatcher():
    path = STATE / 'dispatcher.json'
    write_json(path, {'pid': os.getpid(), 'process_start': process_start(os.getpid()), 'started_at': time.time(),
                      'poll_seconds': 60})
    while True:
        with registry_lock():
            for record in dispatch_once():
                print(json.dumps({'time': time.time(), 'started': record['job'], 'pid': record['pid']}), flush=True)
            snapshot = status()
        if snapshot['all_complete']:
            print(json.dumps({'time': time.time(), 'all_complete': True}), flush=True)
            return 0
        time.sleep(60)


def start_dispatcher():
    previous = STATE / 'dispatcher.json'
    if previous.exists():
        old = read_json(previous)
        if process_start(old['pid']) == old['process_start']:
            return old
    STATE.mkdir(parents=True, exist_ok=True)
    log = (STATE / 'dispatcher.log').open('a')
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--dispatch'], cwd=ROOT,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(100):
        if previous.exists() and read_json(previous)['pid'] == child.pid:
            return read_json(previous)
        time.sleep(0.1)
    raise RuntimeError('Dispatcher did not start; see dispatcher.log')


def drain_requests():
    """Monitor (Codex) files {"job", "reason"} lines; recover here on the host."""
    if not REQUESTS.exists():
        return []
    handled = REQUESTS.with_name(f'recovery_requests.{int(time.time())}.handled.jsonl')
    REQUESTS.replace(handled)
    results = []
    for line in handled.read_text().splitlines():
        request = json.loads(line)
        try:
            with registry_lock():
                record = recover(request['job'], request['reason'])
            results.append({**request, 'started_attempt': record['attempt']})
        except Exception as exc:
            results.append({**request, 'error': f'{type(exc).__name__}: {exc}'})
    with (STATE / 'recoveries.jsonl').open('a') as stream:
        for result in results:
            stream.write(json.dumps({'at': time.time(), **result}) + '\n')
    return results


def first_update(job):
    """A real optimizer update in this job's current attempt log."""
    record = read_json(STATE / job / 'record.json')
    text = Path(record['log']).read_text(errors='replace')
    marker = '"phase": "head_epoch"' if JOBS[job]['module'] == 'crt_kpaths.crt' else '"phase": "train_progress"'
    return marker in text


def confirm_start(wait):
    deadline = time.time() + wait
    while True:
        with registry_lock():
            snapshot = status()
        launched = [job for job in JOBS if (STATE / job / 'record.json').exists()]
        confirmed = {job: first_update(job) for job in launched}
        failed = [job for job in launched if snapshot['runs'][job]['state'] in ('failed', 'exhausted')]
        result = {'checked_at': time.time(), 'launched': launched, 'first_update_confirmed': confirmed,
                  'failed': failed, 'dispatcher_live': bool(snapshot['dispatcher'] and snapshot['dispatcher']['live'])}
        if failed or (launched and all(confirmed.values())) or time.time() > deadline:
            write_json(ROOT / 'logs/crt_kpaths/launch_confirmation.json', result)
            return result
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--launch', action='store_true', help='start (or confirm) the detached dispatcher')
    action.add_argument('--status', action='store_true')
    action.add_argument('--recover', choices=JOBS)
    action.add_argument('--drain-requests', action='store_true')
    action.add_argument('--dispatch', action='store_true')
    action.add_argument('--confirm-start', type=int, metavar='SECONDS',
                        help='wait for the first real update of every launched job')
    action.add_argument('--worker', type=Path)
    parser.add_argument('--reason')
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    if args.worker:
        return worker(args.worker)
    if args.dispatch:
        return dispatcher()
    if args.launch:
        cache = CRT / 'evidence/kpaths_cache_check.json'
        if not cache.exists() or not read_json(cache)['path_mask_equals_connected_depth4_reachability']:
            raise RuntimeError('K-Paths cache and its check must be complete before launch')
        result = start_dispatcher()
    elif args.confirm_start:
        result = confirm_start(args.confirm_start)
    elif args.drain_requests:
        result = drain_requests()
    elif args.recover:
        if not args.reason:
            parser.error('--recover requires --reason')
        with registry_lock():
            result = recover(args.recover, args.reason)
    else:
        with registry_lock():
            result = status()
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

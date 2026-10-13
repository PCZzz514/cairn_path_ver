"""Durable two-arm job records, real export verification, bounded recovery.

rareddie_control copy of accelerated_four/jobs.py: isolated state
(logs/rareddie_control/jobs), outputs (rareddie_control/outputs) and trainer
(rareddie_control.train). The reviewed initial launch is the ONLY --fresh run.
A recovery continues this run's own checkpoint with its exact recorded config
(--resume, or --test-only once its schedule is finished); it never restarts
from scratch, never reads DSN/accelerated/GPU3 state, never changes batch size.

CPU controller only. Training children receive exactly their assigned GPU.
No model forward, preparation, smoke run, or credential access here.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'rareddie_control'
STATE = ROOT / 'logs/rareddie_control/jobs'
OUTPUTS = CONTROL / 'outputs'
CONFIGS = CONTROL / 'configs'
APPROVAL = ROOT / 'logs/rareddie_control/review_approval.json'
EVIDENCE = CONTROL / 'evidence/cse_equivalence.json'
REVIEW_DIFF = CONTROL / 'review/implementation.diff'
ARMS = {'single_drugout': 6, 'double_drugout': 7}
NAMESPACE = 'rareddie_control'
MAX_RECOVERIES = 3


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
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
        # /proc comm may contain spaces; field 22 follows the closing ')'.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def busy(arm):
    # The training child inherits this lock. If its wrapper dies, the live
    # training process still prevents a duplicate launch.
    with (STATE / arm / 'job.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def live(record):
    pid = record['pid']
    return (record['process_start'] is not None and process_start(pid) == record['process_start']) or busy(record['arm'])


def checkpoint(output, name):
    import torch
    return torch.load(output / name, map_location='cpu', weights_only=False)


def history(output):
    path = output / 'history.jsonl'
    # Newline-terminated records only; a torn final append is not a record.
    return [json.loads(line) for line in path.read_text().split('\n')[:-1]] if path.exists() else []


def verify_exports(arm):
    """Completion means selected-best metrics AND all frozen test rows exported."""
    import numpy as np
    import pandas as pd
    import yaml
    from accelerated_four.dsn.metrics import classification_metrics
    from rareddie_control.train import top_k_accuracy

    output = OUTPUTS / arm
    required = ['best.pt', 'last.pt', 'test_metrics.json', 'test_predictions.parquet',
                'test_probabilities.npz', 'test_per_class.csv', 'test_confusion.csv',
                'test_config.yaml', 'test_complete.json']
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        return False, {'missing': missing}
    try:
        best = checkpoint(output, 'best.pt')
        last = checkpoint(output, 'last.pt')
        cfg = best['config']
        assert cfg['gpu'] == ARMS[arm] and cfg['protocol'] == arm and cfg['namespace'] == NAMESPACE
        assert Path(cfg['output']) == output
        assert last['config'] == cfg
        assert last['epoch'] >= cfg['epochs'] or last['bad_epochs'] >= cfg['patience']
        # Selected best is the strict fixed85 maximum of the full history.
        assert best['epoch'] <= last['epoch'] and best['best_metric'] == last['best_metric']
        records = history(output)
        assert [r['epoch'] for r in records] == list(range(1, last['epoch'] + 1))
        assert records[-1] == last['metrics'] and records[best['epoch'] - 1] == best['metrics']
        assert best['best_metric'] == records[best['epoch'] - 1]['validation']['macro_f1_85']
        assert all(r['validation']['macro_f1_85'] <= best['best_metric'] for r in records)
        assert all(r['validation']['macro_f1_85'] < best['best_metric'] for r in records[:best['epoch'] - 1])
        marker = read_json(output / 'test_complete.json')
        metrics = read_json(output / 'test_metrics.json')
        assert marker['checkpoint'] == metrics['checkpoint'] == 'best.pt'
        assert marker['selected_epoch'] == metrics['selected_epoch'] == best['epoch']
        assert marker['selected_update'] == best['update']
        assert metrics['selection'] == 'validation only'
        assert metrics['selected_validation_macro_f1_85'] == best['best_metric']
        eval_cfg = yaml.safe_load((output / 'test_config.yaml').read_text())
        assert all(eval_cfg.get(k) == v for k, v in cfg.items())
        assert eval_cfg['evaluation_checkpoint'] == 'best.pt'
        frozen = pd.read_parquet(Path(cfg['prepared']) / cfg['protocol'] / 'ddi.parquet')
        rows = np.flatnonzero(frozen.split.eq('test').to_numpy())
        expected = frozen.iloc[rows]
        prediction = pd.read_parquet(output / 'test_predictions.parquet')
        assert marker['rows'] == metrics['rows'] == len(prediction) == len(expected)
        assert prediction[expected.columns].reset_index(drop=True).equals(expected.reset_index(drop=True))
        with np.load(output / 'test_probabilities.npz', allow_pickle=False) as data:
            probabilities = data['probabilities']
            labels = data['labels']
            assert probabilities.shape == (len(rows), 85)
            assert np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all()
            assert np.allclose(probabilities.sum(1), 1., atol=1e-5)
            assert np.array_equal(data['event_ids'], np.arange(85))
            assert np.array_equal(labels, expected.event_id.to_numpy())
            assert np.array_equal(data['frozen_row_indices'], rows)
            assert np.array_equal(prediction.frozen_row_index.to_numpy(), rows)
            assert np.array_equal(prediction.predicted_event_id.to_numpy(), probabilities.argmax(1))
            assert np.allclose(prediction.predicted_probability.to_numpy(), probabilities.max(1))
        actual, per_class, confusion = classification_metrics(labels, probabilities)
        # Report-only Top-5 from the same exported probabilities.
        actual['top5_accuracy'] = top_k_accuracy(labels, probabilities)
        for key, value in actual.items():
            assert metrics[key] is None if value is None else np.isclose(metrics[key], value, rtol=1e-6, atol=1e-8)
        exported_class = pd.read_csv(output / 'test_per_class.csv')
        assert np.array_equal(exported_class.event_id, np.arange(85))
        for key in per_class.columns:
            assert np.allclose(exported_class[key], per_class[key], equal_nan=True)
        exported_confusion = pd.read_csv(output / 'test_confusion.csv', index_col=0)
        assert np.array_equal(exported_confusion.index.to_numpy(), np.arange(85))
        assert np.array_equal(exported_confusion.columns.astype(int), np.arange(85))
        assert np.array_equal(exported_confusion.to_numpy(), confusion)
        return True, {'selected_epoch': best['epoch'], 'last_epoch': last['epoch'], 'bad_epochs': last['bad_epochs'],
                      'rows': len(rows), 'metrics': actual}
    except Exception as exc:
        return False, {'verification_error': f'{type(exc).__name__}: {exc}'}


def last_log_event(path):
    """Most recent JSON phase line of an attempt log (progress/epoch/test)."""
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
            return {k: event[k] for k in ['phase', 'epoch', 'update', 'epoch_rows', 'epoch_seconds', 'seconds',
                                          'bad_epochs', 'best_validation_macro_f1_85', 'start_epoch'] if k in event}
    return None


def status():
    result = {}
    for arm, gpu in ARMS.items():
        record_path = STATE / arm / 'record.json'
        if not record_path.exists():
            result[arm] = {'state': 'not_launched', 'gpu': gpu}
            continue
        record = read_json(record_path)
        running = live(record)
        # Avoid reading checkpoints while the trainer is atomically publishing
        # its final files. A live process is always left alone.
        complete, evidence = (False, {}) if running else verify_exports(arm)
        records = history(OUTPUTS / arm)
        process_file = Path(record['log']).with_name(f'attempt_{record["attempt"]}_process.json')
        result[arm] = {'state': 'live' if running else 'complete' if complete else
                      'exhausted' if record['recoveries'] >= MAX_RECOVERIES else 'failed',
                      'gpu': gpu, 'pid': record['pid'], 'attempt': record['attempt'],
                      'recoveries': record['recoveries'], 'mode': record['mode'], 'log': record['log'],
                      'config': record['config'], 'evidence': evidence,
                      'train_process': read_json(process_file) if process_file.exists() else None,
                      'history_last_epoch': records[-1]['epoch'] if records else None,
                      'history_best_validation_macro_f1_85': records[-1]['best_validation_macro_f1_85'] if records else None,
                      'history_bad_epochs': records[-1]['bad_epochs'] if records else None,
                      'last_log_event': last_log_event(record['log']),
                      'blocked_reason': record.get('blocked_reason'),
                      'exit': read_json(Path(record['exit_file'])) if Path(record['exit_file']).exists() else None}
    snapshot = {'checked_at': time.time(), 'runs': result,
                'all_complete': all(r['state'] == 'complete' for r in result.values())}
    write_json(STATE / 'status.json', snapshot)
    if snapshot['all_complete']:
        write_json(STATE / 'final_summary.json', snapshot)
    return snapshot


def start(arm, config_path, mode, recoveries=0, reason='reviewed initial launch'):
    directory = STATE / arm
    directory.mkdir(parents=True, exist_ok=True)
    old_path = directory / 'record.json'
    old = read_json(old_path) if old_path.exists() else None
    if old and live(old):
        raise RuntimeError(f'{arm} is live; cannot relaunch')
    if mode not in ('--fresh', '--resume', '--test-only') or (mode == '--fresh') != (old is None and recoveries == 0):
        raise ValueError('--fresh is only the first reviewed launch; recovery uses --resume or --test-only')
    attempt = old['attempt'] + 1 if old else 1
    log = directory / f'attempt_{attempt}.log'
    if log.exists():
        raise RuntimeError(f'Refusing to overwrite {log}')
    config_path = config_path.resolve()
    if not config_path.is_relative_to(ROOT):
        raise ValueError('Config must remain in workspace')
    exit_file = directory / f'attempt_{attempt}_exit.json'
    command = [sys.executable, '-m', 'rareddie_control.train', '--config', str(config_path), mode]
    launch_file = directory / f'attempt_{attempt}.json'
    record = {'arm': arm, 'gpu': ARMS[arm], 'attempt': attempt, 'recoveries': recoveries,
              'started_at': time.time(), 'config': str(config_path), 'mode': mode,
              'reason': reason, 'log': str(log), 'exit_file': str(exit_file), 'command': command}
    write_json(launch_file, record)
    with log.open('x') as stream:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(launch_file)],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    record.update(pid=child.pid, process_start=process_start(child.pid))
    write_json(launch_file, record)
    write_json(old_path, record)
    (directory / 'job.pid').write_text(str(child.pid) + '\n')
    return record


def worker(launch_file):
    record = read_json(launch_file)
    with (STATE / record['arm'] / 'job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(record['gpu']), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT))
        child = subprocess.Popen(record['command'], cwd=ROOT, env=env, pass_fds=(lock.fileno(),))
        write_json(launch_file.with_name(launch_file.stem + '_process.json'),
                   {'pid': child.pid, 'process_start': process_start(child.pid), 'command': record['command']})
        code = child.wait()
        write_json(Path(record['exit_file']), {'exit_code': code, 'finished_at': time.time()})
        return code if code >= 0 else 128 - code


def resume_point(output):
    """(mode, checkpoint name to resume from) for an incomplete run; never resets."""
    last = checkpoint(output, 'last.pt') if (output / 'last.pt').exists() else None
    best = checkpoint(output, 'best.pt') if (output / 'best.pt').exists() else None
    if last is None and best is None:
        raise RuntimeError('No checkpoint of this run yet; --fresh is initial launch only, never restart from scratch')
    # Crash between best and last publication: best.pt is the complete newer
    # epoch (model, AdamW, RNG, update; its history line precedes it).
    source = 'best.pt' if best is not None and (last is None or best['epoch'] > last['epoch']) else 'last.pt'
    state = best if source == 'best.pt' else last
    cfg = state['config']
    # Own state only: never a DSN/accelerated/GPU3 checkpoint.
    if cfg.get('namespace') != NAMESPACE or Path(cfg['output']) != output:
        raise RuntimeError(f'{source} is not this rareddie_control run')
    done = state['epoch'] >= cfg['epochs'] or state['bad_epochs'] >= cfg['patience']
    if done and best is None:
        raise RuntimeError('Training is done but best.pt is missing; do not reset training')
    return ('--test-only' if done else '--resume'), source, state


def recover(arm, reason):
    directory = STATE / arm
    record = read_json(directory / 'record.json')
    if live(record):
        raise RuntimeError('Live job: no recovery, regardless of log age')
    if record['recoveries'] >= MAX_RECOVERIES:
        raise RuntimeError('Recovery budget exhausted; failure remains recorded')
    # Reserve even if verification or preparation raises. Rejected live or
    # verified-complete jobs do not consume a recovery or get relaunched.
    complete = False
    try:
        complete, _ = verify_exports(arm)
    finally:
        if not complete:
            record['recoveries'] += 1
            write_json(directory / 'record.json', record)
    if complete:
        raise RuntimeError('Verified complete: no recovery needed')
    import yaml
    output = OUTPUTS / arm
    mode, source, state = resume_point(output)
    cfg = state['config']
    # Retain preceding checkpoint inodes, history and all attempt logs. Trainer
    # saves via atomic replacement, so later epochs cannot alter these archives.
    archive = output / 'recovery_archive' / str(record['recoveries'])
    archive.mkdir(parents=True, exist_ok=False)
    for name in ['last.pt', 'best.pt', 'history.jsonl', 'config.yaml', 'test_metrics.json', 'test_complete.json']:
        if (output / name).exists():
            if name.endswith('.pt'):
                os.link(output / name, archive / name)
            else:
                shutil.copyfile(output / name, archive / name)
    if source == 'best.pt':
        temporary = output / 'last.pt.recovery'
        os.link(output / 'best.pt', temporary)
        temporary.replace(output / 'last.pt')
    # Coherent history for the resumed checkpoint: a crash after the history
    # append but before publication must not leave a duplicate epoch line.
    records = [r for r in history(output) if r['epoch'] <= state['epoch']]
    if [r['epoch'] for r in records] != list(range(1, state['epoch'] + 1)) or records[-1] != state['metrics']:
        raise RuntimeError('History is incoherent with the resume checkpoint; archived, not relaunched')
    temporary = output / 'history.jsonl.recovery'
    temporary.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    temporary.replace(output / 'history.jsonl')
    config_path = directory / f'recovery_{record["recoveries"]}_config.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(archive / 'recovery.json', {'reason': reason, 'mode': mode, 'resume_from': source,
                                           'resume_epoch': state['epoch'], 'resume_update': state['update'],
                                           'history_records_kept': len(records), 'at': time.time()})
    return start(arm, config_path, mode, record['recoveries'], reason)


def gpu_busy(index):
    """Compute processes on a physical GPU (nvidia-smi index order)."""
    query = lambda *a: subprocess.run(['nvidia-smi', *a, '--format=csv,noheader'], capture_output=True,
                                      text=True, check=True).stdout.split('\n')
    uuid = dict(line.split(', ') for line in query('--query-gpu=index,uuid') if line)[str(index)]
    return [line for line in query('--query-compute-apps=gpu_uuid,pid,used_memory') if line.startswith(uuid)]


def launch_refusals():
    """Initial launch gate: Codex approval newer than code/diff/evidence, passing
    evidence, idle GPU6/7, and both arms without any record or output state."""
    code = [p for p in CONTROL.rglob('*') if p.is_file() and p.suffix in ('.py', '.sh', '.yaml', '.md')
            and 'outputs' not in p.parts and 'evidence' not in p.parts and '.cache' not in p.parts]
    if not APPROVAL.exists():
        yield f'missing Codex review approval {APPROVAL}'
    else:
        approval = read_json(APPROVAL)
        if approval.get('approved') is not True:
            yield 'review_approval.json: approved is not true'
        if 'codex' not in str(approval.get('reviewer', '')).lower():
            yield 'review_approval.json: reviewer must be Codex'
        newest = max(code + [EVIDENCE, REVIEW_DIFF], key=lambda p: p.stat().st_mtime if p.exists() else 0)
        if newest.exists() and APPROVAL.stat().st_mtime < newest.stat().st_mtime:
            yield f'approval is older than {newest}'
    if not EVIDENCE.exists() or read_json(EVIDENCE).get('passed') is not True:
        yield f'evidence {EVIDENCE} missing or not passed'
    elif EVIDENCE.stat().st_mtime < max(p.stat().st_mtime for p in CONTROL.glob('*.py')):
        yield 'evidence is older than rareddie_control code'
    if not REVIEW_DIFF.exists():
        yield f'missing {REVIEW_DIFF} (run rareddie_control/tools/make_review.sh)'
    for arm, gpu in ARMS.items():
        if (STATE / arm / 'record.json').exists():
            yield f'{arm}: existing job record; use --status and bounded --recover'
        existing = [n for n in ['last.pt', 'best.pt', 'history.jsonl', 'test_complete.json']
                    if (OUTPUTS / arm / n).exists()]
        if existing:
            yield f'{arm}: output already has {existing}; --fresh needs an empty output'
        busy_apps = gpu_busy(gpu)
        if busy_apps:
            yield f'{arm}: physical GPU{gpu} is not idle: {busy_apps}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--launch-reviewed', action='store_true')
    action.add_argument('--status', action='store_true')
    action.add_argument('--recover', choices=ARMS)
    action.add_argument('--block', choices=ARMS)
    action.add_argument('--worker', type=Path)
    parser.add_argument('--reason')
    args = parser.parse_args()
    os.chdir(ROOT)
    # Direct script invocation needs this for metrics import, not only -m.
    sys.path.insert(0, str(ROOT))
    if args.worker:
        return worker(args.worker)
    with registry_lock():
        if args.status:
            result = status()
        elif args.block:
            if not args.reason:
                parser.error('--block requires the observed irreparable --reason')
            record_path = STATE / args.block / 'record.json'
            result = read_json(record_path)
            if live(result) or verify_exports(args.block)[0]:
                raise RuntimeError('Cannot block a live or completed job')
            result.update(recoveries=MAX_RECOVERIES, blocked_reason=args.reason, blocked_at=time.time())
            write_json(record_path, result)
        elif args.recover:
            if not args.reason:
                parser.error('--recover requires an observed failure --reason')
            try:
                result = recover(args.recover, args.reason)
            except Exception as exc:
                recorded = read_json(STATE / args.recover / 'record.json')
                with (STATE / 'recovery_errors.jsonl').open('a') as stream:
                    stream.write(json.dumps({'at': time.time(), 'arm': args.recover, 'reason': args.reason,
                                             'recoveries': recorded['recoveries'],
                                             'error': f'{type(exc).__name__}: {exc}'}) + '\n')
                raise
        else:
            # Initial launch is intentionally distinct from recovery: only after
            # Codex approval, fresh, without existing records or outputs.
            refusals = list(launch_refusals())
            if refusals:
                raise RuntimeError('LAUNCH REFUSED:\n  ' + '\n  '.join(refusals))
            result = {arm: start(arm, CONFIGS / f'{arm}.yaml', '--fresh') for arm in ARMS}
            from rareddie_control.status import update_status
            update_status('launched_awaiting_progress_confirmation',
                          launch={arm: {k: r[k] for k in ['pid', 'process_start', 'mode', 'log', 'gpu', 'started_at']}
                                  for arm, r in result.items()})
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Durable four-arm job records, real export verification, bounded recovery.

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
STATE = ROOT / 'logs/server_jobs'
ARMS = {'path_single_drugout': 4, 'path_double_drugout': 5,
        'no_path_single_drugout': 6, 'no_path_double_drugout': 7}
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


def verify_exports(arm):
    """Completion means selected-best metrics AND all frozen test rows exported."""
    import numpy as np
    import pandas as pd
    import yaml
    from cairn_dsn.metrics import classification_metrics

    output = ROOT / 'outputs' / arm
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
        assert cfg['gpu'] == ARMS[arm]
        assert last['config'] == cfg
        assert last['epoch'] >= cfg['epochs'] or last['bad_epochs'] >= cfg['patience']
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
        frozen = pd.read_parquet(ROOT / 'data/prepared' / cfg['protocol'] / 'ddi.parquet')
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
            mask = data['path_mask']
            assert mask.shape == (len(rows), 2 if cfg['with_path'] else 0)
            if cfg['with_path']:
                assert np.array_equal(mask, prediction[['path_v_to_u_present', 'path_u_to_v_present']].to_numpy())
        actual, per_class, confusion = classification_metrics(labels, probabilities)
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
        return True, {'selected_epoch': best['epoch'], 'rows': len(rows), 'metrics': actual}
    except Exception as exc:
        return False, {'verification_error': f'{type(exc).__name__}: {exc}'}


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
        result[arm] = {'state': 'live' if running else 'complete' if complete else
                      'exhausted' if record['recoveries'] >= MAX_RECOVERIES else 'failed',
                      'gpu': gpu, 'pid': record['pid'], 'attempt': record['attempt'],
                      'recoveries': record['recoveries'], 'log': record['log'],
                      'config': record['config'], 'evidence': evidence,
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
    attempt = old['attempt'] + 1 if old else 1
    log = directory / f'attempt_{attempt}.log'
    if log.exists():
        raise RuntimeError(f'Refusing to overwrite {log}')
    config_path = config_path.resolve()
    if not config_path.is_relative_to(ROOT):
        raise ValueError('Config must remain in workspace')
    exit_file = directory / f'attempt_{attempt}_exit.json'
    command = [sys.executable, '-m', 'cairn_dsn.train', '--config', str(config_path)]
    if mode:
        command.append(mode)
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
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(record['gpu']), PYTHONUNBUFFERED='1')
        child = subprocess.Popen(record['command'], cwd=ROOT, env=env, pass_fds=(lock.fileno(),))
        write_json(launch_file.with_name(launch_file.stem + '_process.json'),
                   {'pid': child.pid, 'process_start': process_start(child.pid), 'command': record['command']})
        code = child.wait()
        write_json(Path(record['exit_file']), {'exit_code': code, 'finished_at': time.time()})
        return code if code >= 0 else 128 - code


def recover(arm, reason, reduce_batch):
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
    output = ROOT / 'outputs' / arm
    cfg = yaml.safe_load(Path(record['config']).read_text())
    mode = ''
    config_path = Path(record['config'])
    last = checkpoint(output, 'last.pt') if (output / 'last.pt').exists() else None
    best_exists = (output / 'best.pt').exists()
    if last is not None:
        cfg = last['config']
        done = last['epoch'] >= cfg['epochs'] or last['bad_epochs'] >= cfg['patience']
        mode = '--test-only' if done else '--resume'
        if done and not best_exists:
            raise RuntimeError('Training is done but best.pt is missing; do not reset training')
    elif best_exists:
        # Crash between best and last publication: recover that exact epoch,
        # including optimizer and RNG, rather than reset its schedule.
        cfg = checkpoint(output, 'best.pt')['config']
        mode = '--resume'
    if reduce_batch:
        log_text = Path(record['log']).read_text(errors='replace')[-30000:]
        if last is not None or best_exists or 'out of memory' not in log_text.lower():
            raise RuntimeError('Batch reduction allowed only for recorded OOM before ANY checkpoint')
        cfg['batch_size'] = max(1, cfg['batch_size'] // 2)
        if cfg['batch_size'] == 1 and yaml.safe_load(config_path.read_text())['batch_size'] == 1:
            raise RuntimeError('Cannot reduce batch below one')
    # Retain preceding checkpoint inodes and all attempt logs. Trainer saves
    # via atomic replacement, so later epochs cannot alter these archives.
    archive = output / 'recovery_archive' / str(record['recoveries'])
    archive.mkdir(parents=True, exist_ok=False)
    for name in ['last.pt', 'best.pt', 'config.yaml', 'test_metrics.json', 'test_complete.json']:
        if (output / name).exists():
            if name.endswith('.pt'):
                os.link(output / name, archive / name)
            else:
                shutil.copyfile(output / name, archive / name)
    if last is None and best_exists:
        os.link(output / 'best.pt', output / 'last.pt')
    config_path = directory / f'recovery_{record["recoveries"]}_config.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return start(arm, config_path, mode, record['recoveries'], reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--launch-reviewed', action='store_true')
    action.add_argument('--status', action='store_true')
    action.add_argument('--recover', choices=ARMS)
    action.add_argument('--block', choices=ARMS)
    action.add_argument('--worker', type=Path)
    parser.add_argument('--reason')
    parser.add_argument('--reduce-batch', action='store_true')
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
                result = recover(args.recover, args.reason, args.reduce_batch)
            except Exception as exc:
                recorded = read_json(STATE / args.recover / 'record.json')
                with (STATE / 'recovery_errors.jsonl').open('a') as stream:
                    stream.write(json.dumps({'at': time.time(), 'arm': args.recover, 'reason': args.reason,
                                             'recoveries': recorded['recoveries'],
                                             'error': f'{type(exc).__name__}: {exc}'}) + '\n')
                raise
        else:
            # Initial launch is intentionally distinct from recovery. Refuse
            # any existing record/checkpoint before starting the first job.
            for arm in ARMS:
                if (STATE / arm / 'record.json').exists() or any((ROOT / 'outputs' / arm / n).exists()
                                                               for n in ['last.pt', 'best.pt']):
                    raise RuntimeError('Existing run found: use --status and bounded --recover')
            result = {arm: start(arm, ROOT / 'configs' / f'{arm}.yaml', '') for arm in ARMS}
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

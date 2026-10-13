"""Stable CPU snapshots of the LIVE original arms into accelerated_four/outputs.

Read-only on originals: each checkpoint is opened once and loaded from that
pinned inode (the original trainer publishes by atomic rename), never signals a
process. A snapshot is accepted only if last.pt, best.pt, history.jsonl and
config.yaml are mutually coherent at the cutoff epoch = last.pt epoch;
otherwise (a concurrent publication) the next consistent state is awaited.

Everything is preserved exactly (model, AdamW, RNG, epoch, update,
best_metric, bad_epochs, metrics record) except the whitelisted configuration
entries output/generator/namespace/optimization. Written per arm:
  snapshot/{last.pt,best.pt,history.jsonl,config.yaml,provenance.json,...}  immutable (0444)
  last.pt, best.pt (hard links; the trainer replaces by rename), history.jsonl, config.yaml (copies)
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accelerated_four.dsn.train import GENERATOR, NAMESPACE, OPTIMIZATION, OUTPUTS, workspace_path  # noqa: E402
from accelerated_four.status import update_status  # noqa: E402

ORIGINAL = ROOT / 'outputs'
ARMS = {'path_single_drugout': 4, 'path_double_drugout': 5,
        'no_path_single_drugout': 6, 'no_path_double_drugout': 7}
WHITELIST = ('output', 'generator', 'namespace', 'optimization')
MISSING = '<absent>'


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def load_pinned(path):
    with path.open('rb') as stream:
        info = os.fstat(stream.fileno())
        value = torch.load(stream, map_location='cpu', weights_only=False)
    return value, {'path': str(path), 'inode': info.st_ino, 'size': info.st_size, 'mtime': info.st_mtime}


def differences(a, b, where=''):
    """Exact recursive comparison; tensors by dtype, shape, device and values."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        same = (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and a.device == b.device and torch.equal(a, b))
        return [] if same else [where]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        same = (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                and np.array_equal(a, b))
        return [] if same else [where]
    if isinstance(a, dict):
        if not isinstance(b, dict) or list(a) != list(b):
            return [where]
        return [d for k in a for d in differences(a[k], b[k], f'{where}/{k}')]
    if isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return [where]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f'{where}[{i}]')]
    return [] if type(a) is type(b) and a == b else [where]


def counts(value):
    if isinstance(value, torch.Tensor):
        return {'tensors': 1, 'elements': value.numel()}
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        parts = [counts(v) for v in value]
        return {'tensors': sum(p['tensors'] for p in parts), 'elements': sum(p['elements'] for p in parts)}
    return {'tensors': 0, 'elements': 0}


def migrate(config, arm):
    migrated = dict(config)
    migrated.update({'output': str(OUTPUTS / arm), 'generator': GENERATOR, 'namespace': NAMESPACE,
                     'optimization': OPTIMIZATION})
    changed = sorted(k for k in set(config) | set(migrated) if config.get(k, MISSING) != migrated.get(k, MISSING))
    assert set(changed) <= set(WHITELIST), changed
    return migrated, {k: {'original': config.get(k, MISSING), 'accelerated': migrated[k]} for k in WHITELIST}


def expected_resume_config(arm, saved_config):
    """The configuration the new trainer builds on --resume (must equal last.pt)."""
    config = yaml.safe_load((ROOT / 'accelerated_four/configs' / f'{arm}.yaml').read_text())
    config.update({'prepared': str(workspace_path(config['prepared'])), 'output': str(workspace_path(config['output'])),
                   'generator': GENERATOR, 'namespace': NAMESPACE, 'optimization': OPTIMIZATION,
                   'selection_metric': 'fixed85 validation macroF1', 'test_selection': False,
                   'source_provenance': json.loads((ROOT / 'docs/source_provenance.json').read_text())})
    config['planned_updates'] = saved_config['planned_updates']
    config['kl_warmup_updates'] = saved_config['kl_warmup_updates']
    return config


def coherence(arm, last, best, lines, config_file):
    """Return a list of incoherence reasons (empty means a consistent snapshot)."""
    reasons = []
    cutoff = last['epoch']
    records = [json.loads(line) for line in lines]
    kept = [r for r in records if r['epoch'] <= cutoff]
    if [r['epoch'] for r in kept] != list(range(1, cutoff + 1)):
        return [f'history epochs {[r["epoch"] for r in kept][-5:]} are not exactly 1..{cutoff}']
    if kept[-1] != last['metrics']:
        reasons.append('last.pt metrics differ from history record at cutoff')
    if kept[-1]['update'] != last['update']:
        reasons.append('last.pt update differs from history')
    # Replay the strict fixed85 selection rule over the retained history.
    best_metric, bad_epochs, best_epoch = -1., 0, None
    for record in kept:
        improved = record['validation']['macro_f1_85'] > best_metric
        if improved:
            best_metric, bad_epochs, best_epoch = record['validation']['macro_f1_85'], 0, record['epoch']
        else:
            bad_epochs += 1
        if (record['improved'] != improved or record['bad_epochs'] != bad_epochs
                or record['best_validation_macro_f1_85'] != best_metric):
            reasons.append(f'history selection fields inconsistent at epoch {record["epoch"]}')
    if best['epoch'] > cutoff:
        reasons.append(f'best.pt epoch {best["epoch"]} is newer than last.pt epoch {cutoff} (publication race)')
    elif best['epoch'] != best_epoch:
        reasons.append(f'best.pt epoch {best["epoch"]} != strict selection epoch {best_epoch}')
    else:
        selected = kept[best_epoch - 1]
        if best['metrics'] != selected or best['update'] != selected['update'] or best['bad_epochs'] != 0:
            reasons.append('best.pt record/update/bad_epochs differ from its history record')
    if not (best['best_metric'] == last['best_metric'] == best_metric):
        reasons.append('best_metric differs between best.pt, last.pt and strict history replay')
    if last['bad_epochs'] != bad_epochs:
        reasons.append('last.pt bad_epochs differs from history replay')
    if not (config_file == last['config'] == best['config']):
        reasons.append('config.yaml, last.pt config and best.pt config differ')
    original = yaml.safe_load((ROOT / 'configs' / f'{arm}.yaml').read_text())
    for key, value in original.items():
        expected = str(workspace_path(value)) if key in ('prepared', 'output') else value
        if last['config'].get(key) != expected:
            reasons.append(f'checkpoint config {key} differs from configs/{arm}.yaml')
    if last['config']['gpu'] != ARMS[arm]:
        reasons.append('GPU mapping differs')
    return reasons


def read_consistent(arm, max_wait, interval):
    source = ORIGINAL / arm
    attempts, deadline = [], time.time() + max_wait
    while True:
        last, last_info = load_pinned(source / 'last.pt')
        best, best_info = load_pinned(source / 'best.pt')
        raw = (source / 'history.jsonl').read_bytes()
        # Only newline-terminated records; an in-progress append is excluded.
        lines = raw.split(b'\n')[:-1]
        config_raw = (source / 'config.yaml').read_bytes()
        reasons = coherence(arm, last, best, lines, yaml.safe_load(config_raw))
        attempts.append({'at': now(), 'last_epoch': last['epoch'], 'best_epoch': best['epoch'],
                         'history_records': len(lines), 'incoherent': reasons})
        if not reasons:
            cutoff_lines = [line for line in lines if json.loads(line)['epoch'] <= last['epoch']]
            return last, best, cutoff_lines, config_raw, {'last.pt': last_info, 'best.pt': best_info}, attempts
        if time.time() > deadline:
            raise RuntimeError(f'{arm}: no consistent snapshot within {max_wait}s: {attempts}')
        print(f'{arm}: incoherent ({reasons}); retrying in {interval}s', flush=True)
        del last, best
        time.sleep(interval)


def live_original(arm):
    attempt = json.loads((ROOT / 'logs/server_jobs' / arm / 'record.json').read_text())['attempt']
    record = json.loads((ROOT / 'logs/server_jobs' / arm / f'attempt_{attempt}_process.json').read_text())
    try:
        start = Path(f'/proc/{record["pid"]}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        start = None
    return {'attempt': attempt, 'pid': record['pid'], 'alive_same_process': start == record['process_start']}


def snapshot(arm, max_wait, interval):
    target = OUTPUTS / arm
    frozen = target / 'snapshot'
    if frozen.exists() or (target / 'last.pt').exists():
        raise RuntimeError(f'{target} already holds a snapshot/run; refusing to overwrite')
    original_process = live_original(arm)
    last, best, lines, config_raw, files, attempts = read_consistent(arm, max_wait, interval)
    migrated, migration = migrate(last['config'], arm)
    resume_config = expected_resume_config(arm, migrated)
    resume_mismatch = sorted(k for k in set(resume_config) | set(migrated)
                             if resume_config.get(k, MISSING) != migrated.get(k, MISSING))
    if resume_mismatch:
        raise RuntimeError(f'{arm}: new trainer resume config would differ on {resume_mismatch}')
    frozen.mkdir(parents=True)
    torch.save({**last, 'config': migrated}, frozen / 'last.pt')
    torch.save({**best, 'config': migrated}, frozen / 'best.pt')
    (frozen / 'history.jsonl').write_bytes(b''.join(line + b'\n' for line in lines))
    (frozen / 'config.yaml').write_text(yaml.safe_dump(migrated, sort_keys=False))
    (frozen / 'original_config.yaml').write_bytes(config_raw)
    for name in ['first_update_gradients.json', 'model_inventory.json']:
        shutil.copyfile(ORIGINAL / arm / name, frozen / f'original_{name}')
    # Exact state equivalence of what was written versus the original objects.
    equivalence = {}
    for name, original in [('last.pt', last), ('best.pt', best)]:
        written = torch.load(frozen / name, map_location='cpu', weights_only=False)
        config_diff = sorted(k for k in set(original['config']) | set(written['config'])
                             if original['config'].get(k, MISSING) != written['config'].get(k, MISSING))
        other = {k: v for k, v in original.items() if k != 'config'}
        mismatches = differences(other, {k: v for k, v in written.items() if k != 'config'})
        equivalence[name] = {'keys': list(written), 'non_config_mismatches': mismatches,
                             'config_keys_changed': config_diff, 'config_equals_migrated': written['config'] == migrated,
                             'model': counts(original['model']), 'optimizer_state': counts(original['optimizer']['state']),
                             'optimizer_param_groups_equal': written['optimizer']['param_groups'] == original['optimizer']['param_groups'],
                             'rng_streams': sorted(original['rng']), 'cuda_rng_devices': len(original['rng']['cuda']),
                             'passed': not mismatches and set(config_diff) <= set(WHITELIST)
                             and written['config'] == migrated}
    history_records = [json.loads(line) for line in lines]
    cfg = migrated
    provenance = {
        'arm': arm, 'gpu': ARMS[arm], 'created_at': now(), 'tool': str(Path(__file__).resolve()),
        'original_output': str(ORIGINAL / arm), 'accelerated_output': str(target),
        'original_files_read': {**files, 'history.jsonl': str(ORIGINAL / arm / 'history.jsonl'),
                                'config.yaml': str(ORIGINAL / arm / 'config.yaml')},
        'original_process_at_snapshot': original_process,
        'read_attempts': attempts,
        'cutoff_epoch': last['epoch'], 'cutoff_update': last['update'],
        'history_records_retained': len(lines), 'history_epochs': [history_records[0]['epoch'], history_records[-1]['epoch']],
        'best_epoch': best['epoch'], 'best_update': best['update'], 'best_metric': best['best_metric'],
        'bad_epochs': last['bad_epochs'], 'epochs': cfg['epochs'], 'patience': cfg['patience'],
        'planned_updates': cfg['planned_updates'], 'kl_warmup_updates': cfg['kl_warmup_updates'],
        'training_done': last['epoch'] >= cfg['epochs'] or last['bad_epochs'] >= cfg['patience'],
        'early_stopped': last['bad_epochs'] >= cfg['patience'],
        'next_epoch': last['epoch'] + 1,
        'config_migration_whitelist': list(WHITELIST), 'config_migration': migration,
        'resume_config_equals_snapshot_config': True,
        'exact_state_equivalence': equivalence,
        'passed': all(v['passed'] for v in equivalence.values())}
    (frozen / 'provenance.json').write_text(json.dumps(provenance, indent=1, allow_nan=False) + '\n')
    if not provenance['passed']:
        raise RuntimeError(f'{arm}: written snapshot is not exactly equivalent: {equivalence}')
    for path in frozen.iterdir():
        path.chmod(0o444)
    # Working state: checkpoints are replaced by rename (links stay intact);
    # history is appended and config.yaml rewritten in place, so those are copies.
    os.link(frozen / 'last.pt', target / 'last.pt')
    os.link(frozen / 'best.pt', target / 'best.pt')
    shutil.copyfile(frozen / 'history.jsonl', target / 'history.jsonl')
    shutil.copyfile(frozen / 'config.yaml', target / 'config.yaml')
    for path in [target / 'history.jsonl', target / 'config.yaml']:
        path.chmod(0o644)
    return {k: provenance[k] for k in ['cutoff_epoch', 'cutoff_update', 'best_epoch', 'best_update', 'best_metric',
                                       'bad_epochs', 'training_done', 'passed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', choices=list(ARMS), default=list(ARMS))
    parser.add_argument('--max-wait', type=float, default=3600.)
    parser.add_argument('--interval', type=float, default=20.)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit("Run on CPU only: CUDA_VISIBLE_DEVICES=''")
    for arm in args.arms:
        update_status('snapshots_running', snapshot_running=arm)
        print(json.dumps({arm: snapshot(arm, args.max_wait, args.interval)}), flush=True)
    done = {}
    for arm in ARMS:
        path = OUTPUTS / arm / 'snapshot/provenance.json'
        if path.exists():
            p = json.loads(path.read_text())
            done[arm] = {k: p[k] for k in ['cutoff_epoch', 'cutoff_update', 'best_epoch', 'best_metric',
                                           'bad_epochs', 'training_done', 'passed']}
    update_status('snapshots_complete' if len(done) == len(ARMS) else 'snapshots_partial',
                  snapshot_running=None, snapshots=done)


if __name__ == '__main__':
    main()

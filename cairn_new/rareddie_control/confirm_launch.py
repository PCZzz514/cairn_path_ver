"""Confirm real progress of both launched arms (no signals, read-only).

Per arm: wrapper/trainer PID, PPID, session, process identity; the fresh start
line; the actual first update (finite CE/loss and gradients, first_update.json),
the parameter inventory and initial RNG record; and a 100-update progress line.
Writes logs/rareddie_control/launch.json and updates status.json.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rareddie_control.jobs import ARMS, OUTPUTS, STATE, live, process_start, read_json, write_json  # noqa: E402
from rareddie_control.status import update_status  # noqa: E402


def events(log):
    found = {}
    for line in Path(log).read_text(errors='replace').splitlines():
        if line.startswith('{'):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            phase = event.get('phase', 'epoch' if 'validation' in event else None)
            if phase == 'train_progress' and 'first_step' not in found:
                found['first_step'] = event
            if phase == 'train_progress' and event['update'] % 100 == 0:
                found['progress_100'] = event
            if phase:
                found[phase] = event
    return found


def ps(pid):
    out = subprocess.run(['ps', '-o', 'pid=,ppid=,pgid=,sid=,etime=,args=', '-p', str(pid)],
                         capture_output=True, text=True).stdout.strip()
    return out or None


def arm_state(arm):
    record = read_json(STATE / arm / 'record.json')
    process = STATE / arm / f'attempt_{record["attempt"]}_process.json'
    trainer = read_json(process) if process.exists() else None
    found = events(record['log'])
    output = OUTPUTS / arm
    first = read_json(output / 'first_update.json') if (output / 'first_update.json').exists() else None
    inventory = read_json(output / 'model_inventory.json') if (output / 'model_inventory.json').exists() else None
    first_ok = bool(first) and first['all_gradients_finite'] and all(
        math.isfinite(v) for v in [*first['loss_terms'].values(), *first['gradient_norms'].values()]) and all(
        v > 0 for v in first['gradient_norms'].values()) and sorted(first['trainable_without_gradient']) == [
        'cse.blocks.1.norm2.bias', 'cse.blocks.1.norm2.weight']
    inventory_ok = bool(inventory) and not inventory['forbidden_name_matches'] and not (
        inventory['dsn_inter_attention_present'] or inventory['inter_projection_present']
        or inventory['path_modules_present']) and len(inventory['molecular_transformer_convs']) == 4
    fresh = (found.get('start') or {}).get('mode') == 'fresh' or record['mode'] != '--fresh'
    ok = 'start' in found and fresh and (('first_step' in found and 'progress_100' in found and first_ok
                                          and inventory_ok) or 'best_checkpoint_test' in found)
    return {'pid': record['pid'], 'wrapper_ps': ps(record['pid']),
            'wrapper_ppid_is_1': (ps(record['pid']) or '').split()[1:2] == ['1'],
            'trainer': trainer, 'trainer_ps': ps(trainer['pid']) if trainer else None,
            'trainer_alive_same_process': bool(trainer) and process_start(trainer['pid']) == trainer['process_start'],
            'live': live(record), 'mode': record['mode'], 'log': record['log'],
            'start': found.get('start'), 'first_step': found.get('first_step'), 'progress_100': found.get('progress_100'),
            'first_update': first, 'first_update_ok': first_ok, 'inventory_ok': inventory_ok,
            'initial_rng_recorded': (output / 'initial_rng.pt').is_file(),
            'test_export': found.get('best_checkpoint_test', {}).get('macro_f1_85'),
            'exit': read_json(Path(record['exit_file'])) if Path(record['exit_file']).exists() else None,
            'progress_confirmed': ok}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', type=float, default=0.)
    args = parser.parse_args()
    deadline = time.time() + args.wait
    while True:
        arms = {arm: arm_state(arm) for arm in ARMS}
        done = all(a['progress_confirmed'] or a['exit'] is not None for a in arms.values())
        if done or time.time() > deadline:
            break
        time.sleep(15)
    gpus = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.splitlines()
    launch = {'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'arms': arms,
              'gpu_compute_apps': gpus, 'loadavg': list(__import__('os').getloadavg()),
              'all_progress_confirmed': all(a['progress_confirmed'] for a in arms.values())}
    write_json(ROOT / 'logs/rareddie_control/launch.json', launch)
    update_status('launched_progress_confirmed' if launch['all_progress_confirmed'] else 'launched_progress_unconfirmed',
                  launch_check={'json': str(ROOT / 'logs/rareddie_control/launch.json'),
                                'all_progress_confirmed': launch['all_progress_confirmed'],
                                'pids': {a: v['pid'] for a, v in arms.items()},
                                'trainer_pids': {a: (v['trainer'] or {}).get('pid') for a, v in arms.items()}})
    print(json.dumps(launch, indent=1))
    return 0 if launch['all_progress_confirmed'] else 1


if __name__ == '__main__':
    sys.exit(main())

"""Confirm real progress of both launched arms (no signals, read-only).

Per arm: wrapper/trainer PID, PPID, session, process identity; the fresh seed42
full-path start line; the actual first update (finite CE/rec/KL/loss, logits and
gradients of EVERY group including path, first_update.json), the parameter
inventory (accelerated_four path architecture, packed attention and path keys)
and initial RNG record; and a 100-update progress line.
Writes logs/pairout_cellout/launch.json and updates status.json.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pairout_cellout.jobs import (ARMS, AUTHORIZED_SCHEDULE, AUTHORIZED_TRAIN_BATCHES, OUTPUTS, STATE,  # noqa: E402
                                  live, process_start, read_json, write_json)
from pairout_cellout.dsn.train import GROUPS, SCHEDULE, TRAIN_BATCHES  # noqa: E402
from pairout_cellout.status import update_status  # noqa: E402

# accelerated_four/outputs/path_single_drugout/model_inventory.json (reviewed
# packed full-path model; same global FIN tables): 13522485 / 13522483.
PARAMETERS, TRAINABLE = 13522485, 13522483
REQUIRED_KEYS = ['cse.blocks.0.interAtt.inter.lin_src.weight', 'cse.blocks.1.interAtt.inter.att_dst',
                 'cse.blocks.1.inter_projection.weight', 'path.rel_kg.3.weight', 'path.attn_relation.3.weight',
                 'pvr.encoder.0.weight', 'classifier.4.weight', 'fin_entities.weight', 'fin_relations.weight']


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
    # GPU3 evidence: the last CSE block's norm2 output is never consumed.
    first_ok = bool(first) and first['all_gradients_finite'] and first['logits_finite'] and all(
        math.isfinite(v) for v in [*first['loss_terms'].values(), *first['gradient_norms'].values()]) and sorted(
        first['gradient_norms']) == sorted(GROUPS) and all(v > 0 for v in first['gradient_norms'].values()) and sorted(
        first['trainable_without_gradient']) == ['cse.blocks.1.norm2.bias', 'cse.blocks.1.norm2.weight'] and (
        first['path_batch']['edges'] > 0 and first['path_batch']['pairs_with_cached_path'] > 0)
    inventory_ok = bool(inventory) and inventory['parameters'] == PARAMETERS and (
        inventory['trainable_parameters'] == TRAINABLE) and inventory['fin_trainable'] and (
        inventory['pair_input'] == 514 and inventory['path_depth'] == 4 and inventory['groups']['path'] > 0) and all(
        key in inventory['state_dict_keys'] for key in REQUIRED_KEYS)
    start = found.get('start') or {}
    # Authorized schedule: 150 epochs, patience 15, 1184 batches, 177600 updates, KL warmup 35520.
    fresh = (start.get('mode') == 'fresh' and start.get('seed') == 42 and start.get('with_path') is True and
             start.get('train_batches') == TRAIN_BATCHES == AUTHORIZED_TRAIN_BATCHES == 1184 and
             SCHEDULE == AUTHORIZED_SCHEDULE and start.get('epochs') == SCHEDULE['epochs'] == 150 and
             start.get('patience') == SCHEDULE['patience'] == 15 and
             start.get('planned_updates') == SCHEDULE['planned_updates'] == 177600 and
             start.get('kl_warmup_updates') == SCHEDULE['kl_warmup_updates'] == 35520 and
             start.get('batch_size') == 128) or record['mode'] != '--fresh'
    # The run's written config and the actual KL beta of the first and 100-update
    # lines must follow the 35520-update warmup (a 23680 warmup would differ).
    run_cfg = yaml.safe_load((output / 'config.yaml').read_text()) if (output / 'config.yaml').is_file() else {}
    schedule_ok = all(run_cfg.get(k) == v for k, v in AUTHORIZED_SCHEDULE.items()) and all(
        event is not None and math.isclose(event['beta'], .001 * min(1., event['update'] / 35520), rel_tol=1e-9)
        for event in [found.get('first_step'), found.get('progress_100')]) and (
        found.get('first_step') or {}).get('update') == (start.get('update', 0) + 1)
    ok = 'start' in found and fresh and (('first_step' in found and 'progress_100' in found and first_ok
                                          and inventory_ok and schedule_ok) or 'best_checkpoint_test' in found)
    return {'pid': record['pid'], 'wrapper_ps': ps(record['pid']),
            'wrapper_ppid_is_1': (ps(record['pid']) or '').split()[1:2] == ['1'],
            'trainer': trainer, 'trainer_ps': ps(trainer['pid']) if trainer else None,
            'trainer_alive_same_process': bool(trainer) and process_start(trainer['pid']) == trainer['process_start'],
            'live': live(record), 'mode': record['mode'], 'log': record['log'],
            'start': found.get('start'), 'first_step': found.get('first_step'), 'progress_100': found.get('progress_100'),
            'first_update': first, 'first_update_ok': first_ok, 'inventory_ok': inventory_ok,
            'schedule_ok': schedule_ok,
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
    write_json(ROOT / 'logs/pairout_cellout/launch.json', launch)
    update_status('launched_progress_confirmed' if launch['all_progress_confirmed'] else 'launched_progress_unconfirmed',
                  launch_check={'json': str(ROOT / 'logs/pairout_cellout/launch.json'),
                                'all_progress_confirmed': launch['all_progress_confirmed'],
                                'pids': {a: v['pid'] for a, v in arms.items()},
                                'trainer_pids': {a: (v['trainer'] or {}).get('pid') for a, v in arms.items()}})
    print(json.dumps(launch, indent=1))
    return 0 if launch['all_progress_confirmed'] else 1


if __name__ == '__main__':
    sys.exit(main())

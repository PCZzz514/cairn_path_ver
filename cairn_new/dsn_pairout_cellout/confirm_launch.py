"""Confirm real progress of both launched arms (no signals, read-only).

Per arm: wrapper/trainer PID, PPID, session, process identity; the fresh seed42
start line with the authorized schedule (150 epochs, patience 15, 1184 batches,
177600 planned updates, 35520 KL warmup updates); the actual first update
(finite CE/rec/KL/loss, logits and finite nonzero gradients of every group:
CSE, DSN InterGraphAttention, inter_projection, NAI, PVR, classifier, FIN; DSN
batch with packed cross edges and no path graph, first_update.json; no path
gradient exists or is required); the parameter inventory (DSN GAT/projection
present, path module NONE, no path arrays, mu -> classifier, counts equal to the
CPU inventory in the cohort/copy evidence) and initial RNG record; and a
100-update progress line. GPU6/7 are shared with the live pure controls.
Writes logs/dsn_pairout_cellout/launch.json and updates status.json.
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
from dsn_pairout_cellout.jobs import (ARMS, AUTHORIZED_SCHEDULE, AUTHORIZED_TRAIN_BATCHES, EVIDENCE,  # noqa: E402
                                       OUTPUTS, STATE, live, process_start, read_json, write_json)
from dsn_pairout_cellout.inventory import GROUPS, UNUSED_TRAINABLE  # noqa: E402
from dsn_pairout_cellout.status import update_status  # noqa: E402


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
    first_ok = bool(first) and first['all_gradients_finite'] and first['logits_finite'] and all(
        math.isfinite(v) for v in [*first['loss_terms'].values(), *first['gradient_norms'].values()]) and sorted(
        first['gradient_norms']) == sorted(GROUPS) and all(v > 0 for v in first['gradient_norms'].values()) and sorted(
        first['trainable_without_gradient']) == UNUSED_TRAINABLE and first['cross_edges_present'] and (
        first['cross_edges'] > 0 and not first['path_graph_present'] and first['path_mask_shape'] == [128, 0])
    # The CPU-constructed inventory of the evidence (same assets, same files).
    reference = read_json(EVIDENCE)['inventory']
    inventory_ok = bool(inventory) and inventory['dsn_inter_attention_present'] and len(
        inventory['dsn_inter_attention_gatconvs']) == 2 and all(
        not g['add_self_loops'] for g in inventory['dsn_inter_attention_gatconvs'].values()) and (
        inventory['inter_projection_present'] and inventory['with_path'] is False
        and inventory['path_module'] == 'NONE' and inventory['path_parameters'] == 0
        and not inventory['collator_path_arrays_loaded'] and inventory['fin_trainable']
        and inventory['pair_input'] == 256 and inventory['pvr_presence_dim'] == 0 and inventory['latent'] == 128
        and len(inventory['molecular_transformer_convs']) == 4) and all(
        inventory[k] == reference[k] for k in ['parameters', 'trainable_parameters', 'groups', 'frozen'])
    start = found.get('start') or {}
    fresh = (start.get('mode') == 'fresh' and start.get('seed') == 42 and start.get('with_path') is False and
             start.get('train_batches') == AUTHORIZED_TRAIN_BATCHES == 1184 and start.get('train_rows') == 151508 and
             start.get('epochs') == AUTHORIZED_SCHEDULE['epochs'] == 150 and
             start.get('patience') == AUTHORIZED_SCHEDULE['patience'] == 15 and
             start.get('planned_updates') == AUTHORIZED_SCHEDULE['planned_updates'] == 177600 and
             start.get('kl_warmup_updates') == AUTHORIZED_SCHEDULE['kl_warmup_updates'] == 35520 and
             start.get('batch_size') == 128) or record['mode'] != '--fresh'
    # Written config and the actual KL beta of the first and 100-update lines
    # follow the 35520-update warmup.
    run_cfg = yaml.safe_load((output / 'config.yaml').read_text()) if (output / 'config.yaml').is_file() else {}
    schedule_ok = all(run_cfg.get(k) == v for k, v in AUTHORIZED_SCHEDULE.items()) and all(
        event is not None and math.isclose(event['beta'], .001 * min(1., event['update'] / 35520), rel_tol=1e-9)
        for event in [found.get('first_step'), found.get('progress_100')])
    ok = 'start' in found and fresh and (('first_step' in found and 'progress_100' in found and first_ok
                                          and inventory_ok and schedule_ok) or 'best_checkpoint_test' in found)
    return {'pid': record['pid'], 'wrapper_ps': ps(record['pid']),
            'wrapper_ppid_is_1': (ps(record['pid']) or '').split()[1:2] == ['1'],
            'trainer': trainer, 'trainer_ps': ps(trainer['pid']) if trainer else None,
            'trainer_alive_same_process': bool(trainer) and process_start(trainer['pid']) == trainer['process_start'],
            'live': live(record), 'mode': record['mode'], 'log': record['log'],
            'start': found.get('start'), 'first_step': found.get('first_step'), 'progress_100': found.get('progress_100'),
            'first_update': first, 'first_update_ok': first_ok, 'inventory_ok': inventory_ok,
            'schedule_ok': schedule_ok, 'parameters': (inventory or {}).get('parameters'),
            'inventory_groups': (inventory or {}).get('groups'),
            'path_module': (inventory or {}).get('path_module'),
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
              'note': ('GPU6/GPU7 shared with the live pure RareDDIE controls (never signalled); CPU/GPU '
                       'contention expected, epoch times are not a promise. Same frozen DDI split, independent '
                       'fresh seed42 replicates, NOT true cell-cold DDI generalization.'),
              'all_progress_confirmed': all(a['progress_confirmed'] for a in arms.values())}
    write_json(ROOT / 'logs/dsn_pairout_cellout/launch.json', launch)
    update_status('launched_progress_confirmed' if launch['all_progress_confirmed'] else 'launched_progress_unconfirmed',
                  launch_check={'json': str(ROOT / 'logs/dsn_pairout_cellout/launch.json'),
                                'all_progress_confirmed': launch['all_progress_confirmed'],
                                'pids': {a: v['pid'] for a, v in arms.items()},
                                'trainer_pids': {a: (v['trainer'] or {}).get('pid') for a, v in arms.items()}})
    print(json.dumps(launch, indent=1))
    return 0 if launch['all_progress_confirmed'] else 1


if __name__ == '__main__':
    sys.exit(main())

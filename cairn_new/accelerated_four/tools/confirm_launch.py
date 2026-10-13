"""Confirm real progress of all four launched arms (no signals, read-only).

Per arm: wrapper/trainer PID, PPID, session, process identity; the start line;
the first training step and a 100-update progress line (or, for a finished
snapshot, the selected-best test export). Also records the original jobs.
Writes logs/accelerated_four/launch.json and updates status.json.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accelerated_four.jobs import ARMS, STATE, live, process_start, read_json, write_json  # noqa: E402
from accelerated_four.status import update_status  # noqa: E402


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
    ok = 'start' in found and (('first_step' in found and 'progress_100' in found) or 'best_checkpoint_test' in found)
    return {'pid': record['pid'], 'wrapper_ps': ps(record['pid']),
            'wrapper_ppid_is_1': (ps(record['pid']) or '').split()[1:2] == ['1'],
            'trainer': trainer, 'trainer_ps': ps(trainer['pid']) if trainer else None,
            'trainer_alive_same_process': bool(trainer) and process_start(trainer['pid']) == trainer['process_start'],
            'live': live(record), 'mode': record['mode'], 'log': record['log'],
            'start': found.get('start'), 'first_step': found.get('first_step'), 'progress_100': found.get('progress_100'),
            'test_export': found.get('best_checkpoint_test', {}).get('macro_f1_85'),
            'exit': read_json(Path(record['exit_file'])) if Path(record['exit_file']).exists() else None,
            'progress_confirmed': ok}


def originals():
    result = {}
    for arm in ARMS:
        record = read_json(ROOT / 'logs/server_jobs' / arm / 'record.json')
        process = read_json(ROOT / 'logs/server_jobs' / arm / f'attempt_{record["attempt"]}_process.json')
        result[arm] = {'pid': process['pid'], 'alive_same_process': process_start(process['pid']) == process['process_start'],
                       'exit': read_json(Path(record['exit_file'])) if Path(record['exit_file']).exists() else None}
    return result


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
    launch = {'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'arms': arms, 'original_jobs': originals(),
              'gpu_compute_apps': gpus, 'loadavg': list(__import__('os').getloadavg()),
              'all_progress_confirmed': all(a['progress_confirmed'] for a in arms.values())}
    write_json(ROOT / 'logs/accelerated_four/launch.json', launch)
    update_status('launched_progress_confirmed' if launch['all_progress_confirmed'] else 'launched_progress_unconfirmed',
                  launch_check={'json': str(ROOT / 'logs/accelerated_four/launch.json'),
                                'all_progress_confirmed': launch['all_progress_confirmed'],
                                'pids': {a: v['pid'] for a, v in arms.items()},
                                'trainer_pids': {a: (v['trainer'] or {}).get('pid') for a, v in arms.items()}})
    print(json.dumps(launch, indent=1))
    return 0 if launch['all_progress_confirmed'] else 1


if __name__ == '__main__':
    sys.exit(main())

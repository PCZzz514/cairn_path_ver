"""Deterministic local note per check; notification only on meaningful change.

Writes logs/accelerated_four/monitor/check_note.json every check (quiet) and
appends to notifications.log only when an arm changes state/attempt (completion,
failure, exhaustion, recovery relaunch) or monitoring ends.
"""
import json
from pathlib import Path
import sys
import time

MONITOR = Path(__file__).resolve().parents[2] / 'logs/accelerated_four/monitor'


def main():
    current = json.loads(Path(sys.argv[1]).read_text())
    outcome = sys.argv[2]
    note_path = MONITOR / 'check_note.json'
    previous = json.loads(note_path.read_text()) if note_path.exists() else {'runs': {}, 'outcome': None}
    runs = {arm: {'state': r['state'], 'attempt': r.get('attempt'), 'recoveries': r.get('recoveries'),
                  'history_last_epoch': r.get('history_last_epoch'), 'bad_epochs': r.get('history_bad_epochs'),
                  'best_validation_macro_f1_85': r.get('history_best_validation_macro_f1_85'),
                  'test': r.get('evidence', {}).get('metrics', {}).get('macro_f1_85')}
            for arm, r in current.get('runs', {}).items()}
    events = [f"{arm}: {previous['runs'].get(arm, {}).get('state')} -> {r['state']} (attempt {r['attempt']}, "
              f"epoch {r['history_last_epoch']}, test macroF1 {r['test']})"
              for arm, r in runs.items()
              if (previous['runs'].get(arm, {}).get('state'), previous['runs'].get(arm, {}).get('attempt'))
              != (r['state'], r['attempt'])]
    if current.get('status_error'):
        events.append('status/verifier error; see the check directory before.json.err/after.json.err')
    if outcome != previous.get('outcome') and outcome != 'monitoring':
        events.append(f'monitor outcome: {outcome}')
    note = {'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'outcome': outcome, 'runs': runs,
            'changed': bool(events), 'events': events}
    temporary = note_path.with_name(note_path.name + '.tmp')
    temporary.write_text(json.dumps(note, indent=1) + '\n')
    temporary.replace(note_path)
    if events:
        with (MONITOR / 'notifications.log').open('a') as stream:
            for event in events:
                stream.write(f"{note['checked_at']} {event}\n")
        print('\n'.join(events))


if __name__ == '__main__':
    main()

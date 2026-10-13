"""Merge benchmark stage/progress into logs/gpu3_speed/status.json.

Codex-owned review fields (ready_for_host_launch, review approval) are never
written here; this only records what the benchmark code itself observed.
"""
import json
from pathlib import Path
import time

STATUS = Path(__file__).resolve().parents[3] / 'logs/gpu3_speed/status.json'
CODEX_OWNED = {'ready_for_host_launch', 'review', 'host_launch_authorized'}


def update_status(stage, **fields):
    if CODEX_OWNED & fields.keys():
        raise ValueError(f'Codex-owned status fields: {sorted(CODEX_OWNED & fields.keys())}')
    status = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    benchmark = status.setdefault('benchmark', {})
    benchmark.update(fields)
    benchmark['stage'] = stage
    benchmark['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    status['state'] = stage
    temporary = STATUS.with_name(STATUS.name + '.tmp')
    temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + '\n')
    temporary.replace(STATUS)
    return status

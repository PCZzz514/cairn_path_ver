"""Merge implementation progress into logs/accelerated_four/status.json (shared with Codex)."""
import json
from pathlib import Path
import time

STATUS = Path(__file__).resolve().parents[1] / 'logs/accelerated_four/status.json'


def update_status(stage, **fields):
    status = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    implementation = status.setdefault('implementation', {})
    implementation.update(fields)
    implementation['stage'] = stage
    implementation['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    status['stage'] = stage
    temporary = STATUS.with_name(STATUS.name + '.tmp')
    temporary.write_text(json.dumps(status, indent=1, allow_nan=False) + '\n')
    temporary.replace(STATUS)

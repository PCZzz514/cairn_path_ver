"""Merge implementation/launch/monitor progress into logs/dsn_pairout_cellout/status.json (shared with Codex)."""
import json
from pathlib import Path
import time

STATUS = Path(__file__).resolve().parents[1] / 'logs/dsn_pairout_cellout/status.json'


def update_status(stage, section='implementation', **fields):
    status = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    part = status.setdefault(section, {})
    part.update(fields)
    part['stage'] = stage
    part['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    status['stage'] = stage
    temporary = STATUS.with_name(STATUS.name + '.tmp')
    temporary.write_text(json.dumps(status, indent=1, allow_nan=False) + '\n')
    temporary.replace(STATUS)

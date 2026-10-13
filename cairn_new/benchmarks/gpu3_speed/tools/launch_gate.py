"""Refuse the five-epoch launch unless Codex approval and passing GPU3 evidence
for the CURRENT code exist, and the benchmark output is fresh."""
import json
import os
from pathlib import Path
import subprocess
import sys
from dsn_gpu3.status import update_status

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / 'benchmarks/gpu3_speed'
LOG_DIR = ROOT / 'logs/gpu3_speed'
APPROVAL = LOG_DIR / 'review_approval.json'
EVIDENCE = BENCH / 'evidence/gpu3_evidence.json'
DIFF = BENCH / 'review/implementation.diff'
OUTPUT = BENCH / 'outputs/path_single_drugout'
CODE = [*sorted((BENCH / 'dsn_gpu3').glob('*.py')), BENCH / 'configs/path_single_drugout.yaml',
        BENCH / 'env.sh', BENCH / 'run_benchmark.sh', BENCH / 'tools/summarize.py']


def reasons():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3':
        yield 'CUDA_VISIBLE_DEVICES is not 3'
    if not APPROVAL.exists():
        yield f'missing Codex review approval {APPROVAL}'
    else:
        approval = json.loads(APPROVAL.read_text())
        if approval.get('approved') is not True:
            yield 'review_approval.json: approved is not true'
        if 'codex' not in str(approval.get('reviewer', '')).lower():
            yield 'review_approval.json: reviewer must be Codex'
        if approval.get('reviewed_diff') != str(DIFF) or approval.get('reviewed_evidence') != str(EVIDENCE):
            yield f'review_approval.json must name reviewed_diff={DIFF} and reviewed_evidence={EVIDENCE}'
        if APPROVAL.stat().st_mtime < max(p.stat().st_mtime for p in CODE + [DIFF, EVIDENCE] if p.exists()):
            yield 'approval is older than the current code, diff or evidence'
    if not EVIDENCE.exists():
        yield f'missing GPU3 evidence {EVIDENCE}'
    else:
        evidence = json.loads(EVIDENCE.read_text())
        if evidence.get('all_numerical_checks_passed') is not True or evidence.get('timing_measured') is not True:
            yield 'evidence: numerical checks not all passed or timing not measured'
        if evidence['environment']['gpu_uuid'].removeprefix('GPU-') != os.environ['GPU3_UUID'].removeprefix('GPU-'):
            yield 'evidence was not produced on physical GPU3'
        if EVIDENCE.stat().st_mtime < max(p.stat().st_mtime for p in CODE):
            yield 'evidence is older than the current benchmark code'
    if not DIFF.exists() or DIFF.stat().st_mtime < max(p.stat().st_mtime for p in CODE):
        yield 'review/implementation.diff missing or older than code (run tools/make_review.sh)'
    if (OUTPUT / 'history.jsonl').exists() or (OUTPUT / 'last.pt').exists():
        yield f'benchmark output {OUTPUT} is not fresh'
    pid_file = LOG_DIR / 'benchmark_runner.pid'
    if pid_file.exists() and Path(f'/proc/{pid_file.read_text().strip()}').exists():
        yield f'benchmark runner already alive (pid {pid_file.read_text().strip()})'
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True).stdout
    if os.environ['GPU3_UUID'] in apps:
        yield 'another process already uses GPU3'


def main():
    failures = list(reasons())
    if failures:
        update_status('launch_refused', launch={'refused': failures})
        print('LAUNCH REFUSED:\n  ' + '\n  '.join(failures), file=sys.stderr)
        sys.exit(1)
    print('launch gate passed')


if __name__ == '__main__':
    main()

"""Completion summarizer for the detached five-epoch GPU3 benchmark.

Runs after the training process has exited. Reads only the recorded history;
status is 'complete' only for exit code 0 with exactly epoch_cap full epochs.
"""
import argparse
import json
from pathlib import Path
import time
import yaml
from dsn_gpu3.status import update_status

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / 'benchmarks/gpu3_speed'
LOG_DIR = ROOT / 'logs/gpu3_speed'
HISTORICAL = (1498, 1534)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exit-code', type=int, required=True)
    parser.add_argument('--config', default=str(BENCH / 'configs/path_single_drugout.yaml'))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output = ROOT / config['output']
    history_path = output / 'history.jsonl'
    history = [json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []
    epochs = [record['epoch'] for record in history]
    complete = args.exit_code == 0 and epochs == list(range(1, config['epoch_cap'] + 1))
    rows = [{'epoch': r['epoch'], 'train_seconds': r['train_seconds'], 'validation_seconds': r['validation_seconds'],
             'seconds': r['seconds'], 'validation_macro_f1_85': r['validation']['macro_f1_85'],
             'validation_accuracy': r['validation']['accuracy'], 'validation_ce': r['validation']['ce'],
             'validation_loss': r['validation']['loss'], 'train_ce': r['train']['ce'], 'train_loss': r['train']['loss'],
             'train_rows': r['train_rows'], 'validation_rows': r['validation_rows'], 'update': r['update'],
             'kl_beta_final': r['train']['kl_beta_final'],
             'speed_ratio_vs_historical': [h / r['seconds'] for h in HISTORICAL]} for r in history]
    summary = {'status': 'complete' if complete else 'failed', 'exit_code': args.exit_code,
               'epochs_recorded': epochs, 'epoch_cap': config['epoch_cap'], 'schedule_epochs': config['epochs'],
               'history': str(history_path), 'per_epoch': rows,
               'historical_original_epoch_seconds': list(HISTORICAL),
               'historical_note': 'original per-pair code, path_single_drugout on GPU4 (train+validation seconds)',
               'summarized_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    if rows:
        mean = sum(r['seconds'] for r in rows) / len(rows)
        summary.update({'mean_epoch_seconds': mean, 'total_train_validation_seconds': sum(r['seconds'] for r in rows),
                        'speed_ratio_vs_historical_mean_epoch': [h / mean for h in HISTORICAL]})
    if not complete:
        summary['failure'] = (f'training exit code {args.exit_code}; epochs recorded {epochs}, '
                              f'expected 1..{config["epoch_cap"]}')
    evidence = BENCH / 'evidence/gpu3_evidence.json'
    if evidence.exists():
        summary['same_gpu_step_evidence'] = {
            'path': str(evidence),
            'median_of_block_medians': json.loads(evidence.read_text())['timing']['median_of_block_medians']}
    for directory in [output, LOG_DIR]:
        (directory / 'five_epoch_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        (directory / 'five_epoch_summary.md').write_text(markdown(summary))
    update_status(summary['status'], result={'summary_json': str(LOG_DIR / 'five_epoch_summary.json'),
                                             'summary_markdown': str(LOG_DIR / 'five_epoch_summary.md'),
                                             'exit_code': args.exit_code, 'epochs_recorded': epochs,
                                             'failure': summary.get('failure')})
    print(json.dumps({k: summary[k] for k in ['status', 'exit_code', 'epochs_recorded']}), flush=True)


def markdown(s):
    lines = [f"# GPU3 packed-attention five-epoch benchmark: {s['status']}", '',
             f"Exit code {s['exit_code']}; epochs recorded {s['epochs_recorded']} of cap {s['epoch_cap']} "
             f"(KL schedule from epochs {s['schedule_epochs']}). Summarized {s['summarized_at']}.", '']
    if s.get('failure'):
        lines += [f"**Failure:** {s['failure']}", '']
    lines += ['| epoch | train s | val s | train+val s | val macroF1(85) | val acc | train CE | train loss | historical/this |',
              '|---|---|---|---|---|---|---|---|---|']
    for r in s['per_epoch']:
        lines.append(f"| {r['epoch']} | {r['train_seconds']:.1f} | {r['validation_seconds']:.1f} | {r['seconds']:.1f} | "
                     f"{r['validation_macro_f1_85']:.4f} | {r['validation_accuracy']:.4f} | {r['train_ce']:.4f} | "
                     f"{r['train_loss']:.4f} | {r['speed_ratio_vs_historical'][0]:.2f}-{r['speed_ratio_vs_historical'][1]:.2f}x |")
    if 'mean_epoch_seconds' in s:
        low, high = s['speed_ratio_vs_historical_mean_epoch']
        lines += ['', f"Mean train+validation epoch {s['mean_epoch_seconds']:.1f} s; historical original "
                      f"{s['historical_original_epoch_seconds'][0]}-{s['historical_original_epoch_seconds'][1]} s "
                      f"-> ratio {low:.2f}-{high:.2f}x (different GPU/time; same-GPU step evidence below)."]
    if 'same_gpu_step_evidence' in s:
        lines += ['', 'Same-GPU3 OLD vs NEW step medians (seconds):', '']
        for k, v in s['same_gpu_step_evidence']['median_of_block_medians'].items():
            lines.append(f"- {k}: OLD {v['old_median_seconds']:.4f}, NEW {v['new_median_seconds']:.4f}, "
                         f"OLD/NEW {v['old_over_new']:.2f}")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()

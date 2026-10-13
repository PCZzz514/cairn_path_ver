"""Combine accelerated_four/evidence/<arm>.json into summary.json and summary.md."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accelerated_four.status import update_status  # noqa: E402

OUT = ROOT / 'accelerated_four/evidence'
ARMS = ['path_single_drugout', 'path_double_drugout', 'no_path_single_drugout', 'no_path_double_drugout']


def fmt(value):
    return 'n/a' if value is None else f'{value:.3e}'


def main():
    arms = {arm: json.loads((OUT / f'{arm}.json').read_text()) for arm in ARMS if (OUT / f'{arm}.json').exists()}
    summary = {'summarized_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'arms_present': list(arms),
               'all_passed': len(arms) == len(ARMS) and all(e['passed'] for e in arms.values()),
               'checks': {arm: e['checks'] for arm, e in arms.items()}}
    lines = ['# accelerated_four resume/compatibility evidence', '',
             f"Summarized {summary['summarized_at']}. **All four passed: {summary['all_passed']}**", '',
             'States are the snapshot last.pt/best.pt the new jobs resume (hard links of the immutable '
             '`snapshot/` files). OLD = original `cairn_dsn` (read-only), NEW = `accelerated_four.dsn`.', '',
             '| arm | cutoff epoch | update | best epoch | best val macroF1(85) | bad epochs | will train | first-batch rows | evidence s |',
             '|---|---|---|---|---|---|---|---|---|']
    for arm, e in arms.items():
        r = e['resume']
        lines.append(f"| {arm} | {r['epoch']} | {r['update']} | {r['best_epoch']} | {r['best_metric']:.4f} | "
                     f"{r['bad_epochs']}/{r['patience']} | {r['will_train']} | "
                     f"{r['first_batch_frozen_row_indices'][:3]}... | {e['seconds']:.0f} |")
    lines += ['', '## Checks', '']
    for arm, e in arms.items():
        lines.append(f"- **{arm}** passed {e['passed']}: " + ', '.join(f'{k}={v}' for k, v in e['checks'].items()))
    lines += ['', '## Collator structure (real next-epoch batches, both directions vs PyG default loop rule)', '',
              '| arm | batches | pairs nh<nt | nh>nt | nh=nt | local loops | global-diagonal edges kept | DataLoader batch == collate |',
              '|---|---|---|---|---|---|---|---|']
    for arm, e in arms.items():
        s = e['collator']['structure']
        lines.append(f"| {arm} | {len(s)} | {sum(x['nh_lt_nt'] for x in s)} | {sum(x['nh_gt_nt'] for x in s)} | "
                     f"{sum(x['nh_eq_nt'] for x in s)} | {sum(x['local_loops'] for x in s)} | "
                     f"{sum(x['global_diagonal_non_loop_edges_kept'] for x in s)} | "
                     f"{e['collator']['real_worker_dataloader_first_batch_exactly_equals_new_collate']} |")
    lines += ['', '## No-path GPU numerical (OLD per-pair vs NEW packed, same seed, train mode, full loss)', '',
              'Gate (GPU3-reviewed): forward rel <= 1e-4; input gradients rel <= 1e-3; every parameter-gradient '
              'tensor max_abs <= 1e-3 * ref + 1e-7 * G (G = largest parameter-gradient magnitude of that batch).', '',
              '| arm | GPU | parameters | batch | loss OLD | loss NEW | forward worst rel | forward worst abs | '
              'param grad worst rel (above atol) | atol | below-atol tensors | gate |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, e in arms.items():
        if 'numerical' not in e:
            continue
        n = e['numerical']
        for set_name, value in n['parameter_sets'].items():
            for b in value['per_batch']:
                g = b['gradient_gate']
                w = g['worst_excluding_below_atol']
                lines.append(f"| {arm} | {n['gpu']['nvidia_smi_uuid'][-12:]} | {set_name} | {b['batch_id']} | "
                             f"{b['loss_old']:.8f} | {b['loss_new']:.8f} | {fmt(b['forward_worst_new_vs_old']['worst_rel'])} | "
                             f"{fmt(b['forward_worst_new_vs_old']['worst_abs'])} | {fmt(w['worst_rel'])} ({w['worst_rel_tensor']}) | "
                             f"{fmt(g['atol'])} | {len(g['tensors_below_atol_analytically_zero_candidates'])} | {g['passed']} |")
        for set_name, value in n['parameter_sets'].items():
            lines.append(f"- {arm} {set_name}: OLD-repeat vs OLD noise floor worst abs "
                         f"{fmt(value['worst_old_repeat_vs_old']['worst_abs'])} ({value['worst_old_repeat_vs_old']['worst_abs_tensor']}); "
                         f"NEW vs OLD worst abs {fmt(value['worst_new_vs_old']['worst_abs'])} ({value['worst_new_vs_old']['worst_abs_tensor']})")
        lines.append(f"- {arm} GPU: {json.dumps(n['gpu'])}")
    lines += ['', 'Below-atol tensors are the analytically zero-gradient biases disclosed in the GPU3 review '
              '(TransformerConv lin_key.bias, SAGPooling lin_rel.bias); their raw relative values are roundoff '
              'residue and the OLD code is not reproducible on them either (OLD-repeat). Path arms reuse the '
              'GPU3-reviewed attention/path numerical evidence; their resume/shuffle/collator evidence is CPU.', '']
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (OUT / 'summary.md').write_text('\n'.join(lines))
    update_status('evidence_complete' if summary['all_passed'] else 'evidence_failed',
                  evidence={'summary': str(OUT / 'summary.md'), 'json': str(OUT / 'summary.json'),
                            'all_passed': summary['all_passed'],
                            'failed': {arm: [k for k, v in e['checks'].items() if not v] for arm, e in arms.items()}})
    print(json.dumps({'all_passed': summary['all_passed']}))
    if not summary['all_passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()

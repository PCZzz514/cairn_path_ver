"""Resume/compatibility evidence for one arm, on the snapshot states the new job resumes.

CPU, every arm: exact original (cairn_dsn, read-only) vs accelerated
initialization order (seed, model, AdamW, collator, datasets, DataLoaders,
schedule, strict checkpoint/AdamW load, THEN RNG restore, then iteration),
strict resume config with only whitelisted migration, exact next-epoch
permutation and real DataLoader first batch, exact collator edges.
GPU (no-path arms, physical GPU6 single / GPU7 double): OLD per-pair vs NEW
packed full model on the real next-epoch batches with the snapshot last.pt and
best.pt parameters: logits/CE/rec/KL/loss, eval logits, every parameter and
input gradient, OLD-repeat noise floor, GPU3-reviewed gradient gate.
Writes accelerated_four/evidence/<arm>.json; no training, no checkpoint writes.
"""
import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import add_self_loops, remove_self_loops
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cairn_dsn.data import Collator as OldCollator, DDIDataset as OldDataset  # noqa: E402
from cairn_dsn.model import DSNRareDDI as OldModel  # noqa: E402
from accelerated_four.dsn import train as new_train  # noqa: E402
from accelerated_four.dsn.data import Collator as NewCollator, DDIDataset as NewDataset, to_device  # noqa: E402
from accelerated_four.dsn.model import DSNRareDDI as NewModel  # noqa: E402
from accelerated_four.tools.snapshot import ARMS, MISSING, WHITELIST, differences, expected_resume_config  # noqa: E402

OUT = ROOT / 'accelerated_four/evidence'
NUMERIC_BATCHES = 3
# Identical to the GPU3-reviewed thresholds (benchmarks/gpu3_speed/tools/evidence.py).
TOL = {'model_forward_rel': 1e-4, 'model_grad_rel': 1e-3, 'model_grad_atol_of_global_scale': 1e-7}


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def restore(state):
    """The trainer's restore_rng; without a visible GPU the CUDA stream is checked, not set."""
    if torch.cuda.is_available():
        new_train.restore_rng(state)
        return 'trainer restore_rng (python, numpy, torch, cuda)'
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    return 'python, numpy, torch CPU (no visible GPU; CUDA state compared as saved bytes)'


def cpu_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}


def same_rng(a, b):
    return not differences({k: a[k] for k in ['python', 'numpy', 'torch']}, {k: b[k] for k in ['python', 'numpy', 'torch']})


def permutation(loader, saved_rng):
    """Index order of one epoch with the same RNG draws as iter(train loader)."""
    restore(saved_rng)
    order = DataLoader(range(len(loader.dataset)), batch_size=loader.batch_size, shuffle=True)
    return torch.cat(list(order)).numpy()


def initialize(kind, arm, saved, best):
    """Trainer order for --resume; kind 'old' uses the original cairn_dsn classes."""
    Model, Collator, Dataset = (OldModel, OldCollator, OldDataset) if kind == 'old' else (NewModel, NewCollator, NewDataset)
    config = expected_resume_config(arm, saved['config'])
    torch.set_num_threads(config['cpu_threads'])
    seed_all(config['seed'])
    prepared = Path(config['prepared'])
    assets = torch.load(prepared / 'assets.pt', map_location='cpu', weights_only=False)
    model = Model(assets, config)
    after_construction = cpu_rng()
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    collate = Collator(assets, prepared, config['with_path'])
    datasets = {split: Dataset(prepared, config['protocol'], split) for split in ['train', 'val']}
    loaders = {split: DataLoader(dataset, batch_size=config['batch_size'], shuffle=split == 'train',
                                 num_workers=config['num_workers'], collate_fn=collate, pin_memory=True,
                                 persistent_workers=False)
               for split, dataset in datasets.items()}
    computed_planned = config['epochs'] * len(loaders['train'])
    computed_warmup = max(1, math.ceil(config['kl_warmup_fraction'] * computed_planned))
    strict_config_equal = saved['config'] == config
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    restored_by = restore(saved['rng'])
    after_restore = cpu_rng()
    if torch.cuda.is_available():
        after_restore['cuda_equal_saved'] = not differences(torch.cuda.get_rng_state_all(), saved['rng']['cuda'])
    first = None
    if kind == 'new':
        # The REAL training DataLoader (2 nonpersistent workers, pinned) first batch.
        first = next(iter(loaders['train']))
    loaded_model_mismatch = differences(dict(model.state_dict()), saved['model'])
    loaded_optimizer_mismatch = differences(optimizer.state_dict(), saved['optimizer'])
    best_model = Model(assets, config)
    best_model.load_state_dict(best['model'])
    best_mismatch = differences(dict(best_model.state_dict()), best['model'])
    return {'config': config, 'assets': assets, 'model': model, 'best_model': best_model, 'optimizer': optimizer,
            'collate': collate, 'datasets': datasets, 'loaders': loaders, 'after_construction': after_construction,
            'init_state': init_state, 'after_restore': after_restore, 'restored_by': restored_by,
            'computed_planned': computed_planned, 'computed_warmup': computed_warmup,
            'strict_config_equal': strict_config_equal, 'first': first,
            'loaded_model_mismatch': loaded_model_mismatch, 'loaded_optimizer_mismatch': loaded_optimizer_mismatch,
            'best_model_mismatch': best_mismatch, 'permutation': permutation(loaders['train'], saved['rng'])}


def structure(old_raw, new_raw):
    """Exact per-pair edges against PyG's own default GATConv loop rule (GPU3 reviewed check)."""
    ga, gb, packed = new_raw['graph_a'], new_raw['graph_b'], new_raw['cross_edges']
    hp, tp = ga.ptr.tolist(), gb.ptr.tolist()
    result = {'pairs': len(old_raw['cross_edges']), 'packed_edges': packed.shape[1], 'direction_h_to_t_exact': True,
              'direction_t_to_h_exact': True, 'all_edges_within_own_pair': True, 'local_loops_appended_last': True,
              'local_loops': 0, 'nh_lt_nt': 0, 'nh_gt_nt': 0, 'nh_eq_nt': 0,
              'global_diagonal_non_loop_edges_kept': 0}
    seg = 0
    for i, local in enumerate(old_raw['cross_edges']):
        nh, nt = hp[i + 1] - hp[i], tp[i + 1] - tp[i]
        m = min(nh, nt)
        part = packed[:, seg:seg + nh * nt]
        seg += nh * nt
        offset = torch.tensor([[hp[i]], [tp[i]]])
        expected_ht = add_self_loops(remove_self_loops(local)[0], num_nodes=m)[0]
        expected_th = add_self_loops(remove_self_loops(local[[1, 0]])[0], num_nodes=m)[0]
        result['direction_h_to_t_exact'] &= torch.equal(part - offset, expected_ht)
        result['direction_t_to_h_exact'] &= torch.equal(part[[1, 0]] - offset[[1, 0]], expected_th)
        result['all_edges_within_own_pair'] &= bool(((part[0] >= hp[i]) & (part[0] < hp[i + 1]) &
                                                     (part[1] >= tp[i]) & (part[1] < tp[i + 1])).all())
        loc = part - offset
        tail = loc[:, nh * nt - m:]
        result['local_loops_appended_last'] &= (torch.equal(tail, torch.arange(m).repeat(2, 1)) and
                                                int((loc[0] == loc[1]).sum()) == m)
        result['local_loops'] += m
        result['nh_lt_nt' if nh < nt else 'nh_gt_nt' if nh > nt else 'nh_eq_nt'] += 1
        result['global_diagonal_non_loop_edges_kept'] += int(((part[0] == part[1]) & (loc[0] != loc[1])).sum())
    result['segments_cover_packed'] = seg == packed.shape[1]
    result['passed'] = (result['direction_h_to_t_exact'] and result['direction_t_to_h_exact'] and
                        result['all_edges_within_own_pair'] and result['local_loops_appended_last'] and
                        result['segments_cover_packed'] and result['nh_lt_nt'] > 0 and result['nh_gt_nt'] > 0)
    return result


def batch_equal(a, b):
    """Exact equality of two collated batches (tensors, PyG graphs, path dict)."""
    if isinstance(a, Batch):
        return all(torch.equal(getattr(a, k), getattr(b, k)) for k in ['x', 'edge_index', 'edge_attr', 'batch', 'ptr'])
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(batch_equal(a[k], b[k]) for k in a)
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and torch.equal(a, b)
    return a == b


def discrepancy(new, old):
    new, old = new.detach().double(), old.detach().double()
    assert new.shape == old.shape, (new.shape, old.shape)
    error = (new - old).abs().max().item() if new.numel() else 0.
    ref = old.abs().max().item() if old.numel() else 0.
    rel = error / ref if ref > 0 else (0. if error == 0 else None)
    return {'max_abs': error, 'ref_max_abs': ref, 'rel': rel}


def worst(table):
    rel = max(table, key=lambda k: float('inf') if table[k]['rel'] is None else table[k]['rel'])
    absolute = max(table, key=lambda k: table[k]['max_abs'])
    return {'worst_rel': table[rel]['rel'], 'worst_rel_tensor': rel,
            'worst_abs': table[absolute]['max_abs'], 'worst_abs_tensor': absolute}


def gradient_check(table, noise):
    """GPU3-reviewed gate for every parameter-gradient tensor of ONE batch."""
    scale = max(v['ref_max_abs'] for v in table.values())
    atol = TOL['model_grad_atol_of_global_scale'] * scale
    failures = {k: v for k, v in table.items() if v['max_abs'] > atol + TOL['model_grad_rel'] * v['ref_max_abs']}
    below = {k: {'new_vs_old': v, 'old_repeat_vs_old': noise.get(k)} for k, v in table.items() if v['ref_max_abs'] <= atol}
    above = {k: v for k, v in table.items() if k not in below}
    return {'passed': not failures, 'tensors_checked': len(table), 'global_gradient_scale': scale, 'atol': atol,
            'failures': failures, 'tensors_below_atol_analytically_zero_candidates': below,
            'worst_excluding_below_atol': worst(above) if above else None}


def to_gpu(raw, device):
    return to_device({k: copy.copy(v) if isinstance(v, Batch) else v for k, v in raw.items()}, device)


def model_pass(model, raw, seed, config, beta, device):
    batch = to_gpu(raw, device)
    leaves = {}
    for graph in ['graph_a', 'graph_b']:
        for attr in ['x', 'edge_attr']:
            leaf = getattr(batch[graph], attr).clone().requires_grad_()
            setattr(batch[graph], attr, leaf)
            leaves[f'input.{graph}.{attr}'] = leaf
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    out = model(batch)
    ce = F.cross_entropy(out['logits'], batch['label'])
    loss = ce + config['reconstruction_weight'] * out['rec'] + beta * out['kl']
    loss.backward()
    forward = {'logits': out['logits'].detach(), 'ce': ce.detach(), 'rec': out['rec'].detach(),
               'kl': out['kl'].detach(), 'loss': loss.detach()}
    params = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    inputs = {name: leaf.grad.detach().clone() for name, leaf in leaves.items()}
    model.eval()
    with torch.no_grad():
        torch.manual_seed(seed)
        evaluated = model(to_gpu(raw, device))
    forward['eval.logits'] = evaluated['logits']
    forward['eval.rec'] = evaluated['rec']
    model.zero_grad(set_to_none=True)
    return forward, out['path_mask'].detach(), params, inputs


def model_compare(a, b):
    forward = {k: discrepancy(a[0][k], b[0][k]) for k in a[0]}
    params = {k: discrepancy(a[2][k], b[2][k]) for k in a[2]} if a[2].keys() == b[2].keys() else None
    inputs = {k: discrepancy(a[3][k], b[3][k]) for k in a[3]}
    return {'forward': forward, 'parameters': params, 'inputs': inputs, 'path_mask_equal': torch.equal(a[1], b[1]),
            'gradient_parameter_count': len(a[2])}


def gpu_identity(gpu):
    device = torch.device('cuda:0')
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    smi = run(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid', '--format=csv,noheader'])
    apps = run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader']).splitlines()
    return {'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_device_order': os.environ.get('CUDA_DEVICE_ORDER'), 'torch_device_uuid': uuid,
            'nvidia_smi_uuid': smi, 'physical_gpu_verified': uuid.removeprefix('GPU-') == smi.removeprefix('GPU-'),
            'gpu_name': torch.cuda.get_device_name(device), 'compute_apps_on_this_gpu': [a for a in apps if smi in a],
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'torch': torch.__version__,
            'utilization': run(['nvidia-smi', '-i', str(gpu), '--query-gpu=utilization.gpu,memory.used,memory.total',
                                '--format=csv,noheader']), 'loadavg': os.getloadavg()}


def numerical(old, new, raws_old, raws_new, saved, best, config, gpu):
    device = torch.device('cuda:0')
    identity = gpu_identity(gpu)
    old_model, new_model = old['model'].to(device), new['model'].to(device)
    # The actual schedule value of the first resumed update.
    beta = config['kl_weight'] * min(1., (saved['update'] + 1) / saved['config']['kl_warmup_updates'])
    result = {'gpu': identity, 'beta': beta, 'tolerances': TOL, 'parameter_sets': {}}
    for set_name, state in [('snapshot_last_resumed', saved['model']), ('snapshot_best_exported', best['model'])]:
        old_model.load_state_dict(state, strict=True)
        new_model.load_state_dict(state, strict=True)
        per_batch, tables = [], {'new_vs_old': {}, 'old_repeat_vs_old': {}}
        for b in range(len(raws_new)):
            seed = 1000 + b
            first = model_pass(old_model, raws_old[b], seed, config, beta, device)
            repeat = model_pass(old_model, raws_old[b], seed, config, beta, device)
            packed = model_pass(new_model, raws_new[b], seed, config, beta, device)
            versus, noise = model_compare(packed, first), model_compare(repeat, first)
            gate = (gradient_check(versus['parameters'], noise['parameters'] or {})
                    if versus['parameters'] is not None else {'passed': False, 'failures': 'gradient sets differ'})
            per_batch.append({
                'batch_id': b, 'seed': seed, 'loss_old': first[0]['loss'].item(), 'loss_new': packed[0]['loss'].item(),
                'ce_old': first[0]['ce'].item(), 'rec_old': first[0]['rec'].item(), 'kl_old': first[0]['kl'].item(),
                'forward_new_vs_old': versus['forward'], 'forward_old_repeat_vs_old': noise['forward'],
                'forward_worst_new_vs_old': worst(versus['forward']),
                'forward_worst_old_repeat_vs_old': worst(noise['forward']),
                'input_gradients_worst_new_vs_old': worst(versus['inputs']),
                'input_gradients_worst_old_repeat_vs_old': worst(noise['inputs']),
                'parameter_gradients_worst_new_vs_old': worst(versus['parameters']) if versus['parameters'] else None,
                'parameter_gradients_worst_old_repeat_vs_old': worst(noise['parameters']) if noise['parameters'] else None,
                'path_mask_equal': versus['path_mask_equal'], 'gradient_parameter_count': versus['gradient_parameter_count'],
                'gradient_gate': gate})
            for label, value in [('new_vs_old', versus), ('old_repeat_vs_old', noise)]:
                tables[label].update({f'batch{b}.forward.{k}': v for k, v in value['forward'].items()})
                tables[label].update({f'batch{b}.input.{k}': v for k, v in value['inputs'].items()})
                tables[label].update({f'batch{b}.param.{k}': v for k, v in (value['parameters'] or {}).items()})
        checks = {
            'forward': all(worst(r['forward_new_vs_old'])['worst_rel'] is not None and
                           worst(r['forward_new_vs_old'])['worst_rel'] <= TOL['model_forward_rel'] for r in per_batch),
            'input_gradients': all(r['input_gradients_worst_new_vs_old']['worst_rel'] is not None and
                                   r['input_gradients_worst_new_vs_old']['worst_rel'] <= TOL['model_grad_rel'] for r in per_batch),
            'parameter_gradients_gate': all(r['gradient_gate']['passed'] for r in per_batch),
            'gradient_sets_and_mask': all(r['path_mask_equal'] and r['parameter_gradients_worst_new_vs_old'] is not None
                                          for r in per_batch)}
        result['parameter_sets'][set_name] = {'per_batch': per_batch, 'checks': checks,
                                              'worst_new_vs_old': worst(tables['new_vs_old']),
                                              'worst_old_repeat_vs_old': worst(tables['old_repeat_vs_old'])}
    result['passed'] = all(all(v['checks'].values()) for v in result['parameter_sets'].values())
    old['model'].cpu()
    new['model'].cpu()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=list(ARMS), required=True)
    args = parser.parse_args()
    arm, gpu = args.arm, ARMS[args.arm]
    with_path = not arm.startswith('no_path')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != ('' if with_path else str(gpu)):
        raise SystemExit(f"{arm}: run with CUDA_VISIBLE_DEVICES={'<empty>' if with_path else gpu}")
    started = time.time()
    output = ROOT / 'accelerated_four/outputs' / arm
    provenance = json.loads((output / 'snapshot/provenance.json').read_text())
    saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
    best = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    evidence = {'arm': arm, 'gpu': gpu, 'started_at': now(), 'tool': str(Path(__file__).resolve()),
                'resumed_files': {name: {'path': str(output / name), 'inode': (output / name).stat().st_ino,
                                         'same_inode_as_immutable_snapshot':
                                             (output / name).stat().st_ino == (output / 'snapshot' / name).stat().st_ino}
                                  for name in ['last.pt', 'best.pt']},
                'snapshot_provenance': {k: provenance[k] for k in [
                    'cutoff_epoch', 'cutoff_update', 'best_epoch', 'best_update', 'best_metric', 'bad_epochs',
                    'training_done', 'early_stopped', 'config_migration_whitelist', 'passed']}}

    # Whitelisted migration against the original resolved config (written once at original start).
    original_cfg = yaml.safe_load((ROOT / 'outputs' / arm / 'config.yaml').read_text())
    snapshot_original_cfg = yaml.safe_load((output / 'snapshot/original_config.yaml').read_text())
    changed = sorted(k for k in set(original_cfg) | set(saved['config'])
                     if original_cfg.get(k, MISSING) != saved['config'].get(k, MISSING))
    evidence['config'] = {'keys_changed_vs_original': changed, 'whitelist': list(WHITELIST),
                          'only_whitelist_changed': set(changed) <= set(WHITELIST),
                          'original_config_unchanged_since_snapshot': original_cfg == snapshot_original_cfg,
                          'last_best_config_equal': saved['config'] == best['config'],
                          'working_config_yaml_equal': yaml.safe_load((output / 'config.yaml').read_text()) == saved['config']}

    old = initialize('old', arm, saved, best)
    new = initialize('new', arm, saved, best)
    config = new['config']
    first_rows = new['first']['row_index'].numpy()
    positions = np.searchsorted(new['datasets']['train'].rows, first_rows)
    perm = new['permutation']
    cutoff_history = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines()]
    steps = {float(v['step']) for v in saved['optimizer']['state'].values()}
    group = saved['optimizer']['param_groups'][0]
    evidence['resume'] = {
        'strict_resume_config_equal_new': new['strict_config_equal'],
        'strict_resume_config_equal_old_order': old['strict_config_equal'],
        'state_dict_names_equal_old_new': list(old['init_state']) == list(new['init_state']),
        'state_dict_shapes_equal_old_new': all(old['init_state'][k].shape == new['init_state'][k].shape for k in old['init_state']),
        'state_dict_names_equal_checkpoint': list(new['init_state']) == list(saved['model']),
        'state_dict_entries': len(saved['model']),
        'top_level_modules': sorted({k.split('.')[0] for k in saved['model']}),
        'seed42_construction_bitwise_equal_old_new': not differences(old['init_state'], new['init_state']),
        'cpu_rng_after_construction_equal_old_new': same_rng(old['after_construction'], new['after_construction']),
        'strict_model_load_exact_new': not new['loaded_model_mismatch'],
        'strict_model_load_exact_old': not old['loaded_model_mismatch'],
        'adamw_state_load_exact_new': not new['loaded_optimizer_mismatch'],
        'adamw_state_load_exact_old': not old['loaded_optimizer_mismatch'],
        'best_model_strict_load_exact': not new['best_model_mismatch'] and not old['best_model_mismatch'],
        'adamw_param_group': {k: v for k, v in group.items() if k != 'params'},
        'adamw_hyperparameters_equal_config': group['lr'] == config['learning_rate'] and group['weight_decay'] == config['weight_decay'],
        'adamw_state_steps': sorted(steps), 'adamw_steps_equal_update': steps == {float(saved['update'])},
        'adamw_state_tensors': len(saved['optimizer']['state']),
        'rng_restored_by': new['restored_by'],
        'cpu_rng_after_restore_equals_saved': same_rng(new['after_restore'], saved['rng']) and same_rng(old['after_restore'], saved['rng']),
        'cuda_rng_saved_devices': len(saved['rng']['cuda']),
        'cuda_rng_after_restore_equals_saved': new['after_restore'].get('cuda_equal_saved', 'no visible GPU (CPU evidence)'),
        'epoch': saved['epoch'], 'update': saved['update'], 'start_epoch': saved['epoch'] + 1,
        'train_batches': len(new['loaders']['train']), 'update_equals_epoch_x_batches': saved['update'] == saved['epoch'] * len(new['loaders']['train']),
        'planned_updates_saved': saved['config']['planned_updates'], 'planned_updates_recomputed': new['computed_planned'],
        'kl_warmup_saved': saved['config']['kl_warmup_updates'], 'kl_warmup_recomputed': new['computed_warmup'],
        'schedule_equal': (saved['config']['planned_updates'] == new['computed_planned'] == 116500 and
                           saved['config']['kl_warmup_updates'] == new['computed_warmup'] == 23300),
        'beta_first_resumed_update': config['kl_weight'] * min(1., (saved['update'] + 1) / saved['config']['kl_warmup_updates']),
        'epochs': config['epochs'], 'patience': config['patience'], 'bad_epochs': saved['bad_epochs'],
        'best_metric': saved['best_metric'], 'best_epoch': best['epoch'],
        'will_train': saved['bad_epochs'] < config['patience'] and saved['epoch'] < config['epochs'],
        'history_records': len(cutoff_history), 'history_last_epoch_equals_cutoff': cutoff_history[-1]['epoch'] == saved['epoch'],
        'history_last_record_equals_checkpoint_metrics': cutoff_history[-1] == saved['metrics'],
        'best_record_equals_history': cutoff_history[best['epoch'] - 1] == best['metrics'],
        'next_epoch_permutation_equal_old_new': bool(np.array_equal(old['permutation'], perm)),
        'permutation_is_full_train_permutation': bool(np.array_equal(np.sort(perm), np.arange(len(new['datasets']['train'])))),
        'real_dataloader_first_batch_equals_permutation_head': bool(np.array_equal(positions, perm[:config['batch_size']])),
        'first_batch_frozen_row_indices': first_rows.tolist()}

    # Real next-epoch batches (the first equals the real worker DataLoader batch), both collators.
    train = new['datasets']['train']
    raws_old, raws_new, structures = [], [], []
    for b in range(NUMERIC_BATCHES):
        rows = [train[i] for i in perm[b * config['batch_size']:(b + 1) * config['batch_size']]]
        raws_old.append(old['collate'](rows))
        raws_new.append(new['collate'](rows))
        structures.append({'batch_id': b, **structure(raws_old[-1], raws_new[-1])})
    first_cpu = {k: v for k, v in new['first'].items()}
    evidence['collator'] = {
        'batches': NUMERIC_BATCHES, 'batch_size': config['batch_size'],
        'real_worker_dataloader_first_batch_exactly_equals_new_collate': batch_equal(first_cpu, raws_new[0]),
        'old_new_labels_rows_graphs_equal': all(
            torch.equal(o['label'], n['label']) and torch.equal(o['row_index'], n['row_index']) and
            batch_equal(o['graph_a'], n['graph_a']) and batch_equal(o['graph_b'], n['graph_b']) and
            (not with_path or batch_equal(o['path_graph'], n['path_graph'])) for o, n in zip(raws_old, raws_new)),
        'structure': structures, 'structure_passed': all(s['passed'] for s in structures)}

    if not with_path:
        evidence['numerical'] = numerical(old, new, raws_old, raws_new, saved, best, config, gpu)

    r, c = evidence['resume'], evidence['collator']
    checks = {
        'snapshot_provenance_passed': provenance['passed'],
        'resumed_files_are_immutable_snapshot': all(v['same_inode_as_immutable_snapshot'] for v in evidence['resumed_files'].values()),
        'config_only_whitelist_changed': evidence['config']['only_whitelist_changed'] and evidence['config']['last_best_config_equal']
        and evidence['config']['working_config_yaml_equal'] and evidence['config']['original_config_unchanged_since_snapshot'],
        'strict_resume_config': r['strict_resume_config_equal_new'] and r['strict_resume_config_equal_old_order'],
        'state_dict_layout': r['state_dict_names_equal_old_new'] and r['state_dict_shapes_equal_old_new'] and r['state_dict_names_equal_checkpoint'],
        'construction_and_rng': r['seed42_construction_bitwise_equal_old_new'] and r['cpu_rng_after_construction_equal_old_new'],
        'strict_model_adamw_load_exact': r['strict_model_load_exact_new'] and r['strict_model_load_exact_old']
        and r['adamw_state_load_exact_new'] and r['adamw_state_load_exact_old'] and r['best_model_strict_load_exact'],
        'adamw_hyperparameters_and_steps': r['adamw_hyperparameters_equal_config'] and r['adamw_steps_equal_update'],
        'rng_restore_and_permutation': r['cpu_rng_after_restore_equals_saved'] and r['next_epoch_permutation_equal_old_new']
        and r['cuda_rng_after_restore_equals_saved'] is not False
        and r['permutation_is_full_train_permutation'] and r['real_dataloader_first_batch_equals_permutation_head'],
        'epoch_update_schedule': r['update_equals_epoch_x_batches'] and r['schedule_equal'],
        'history_and_selection': r['history_last_epoch_equals_cutoff'] and r['history_last_record_equals_checkpoint_metrics']
        and r['best_record_equals_history'] and saved['best_metric'] == best['best_metric'],
        'collator_exact': c['real_worker_dataloader_first_batch_exactly_equals_new_collate'] and c['old_new_labels_rows_graphs_equal']
        and c['structure_passed']}
    if not with_path:
        checks['no_path_gpu_numerical'] = evidence['numerical']['passed'] and evidence['numerical']['gpu']['physical_gpu_verified']
    evidence['checks'] = checks
    evidence['passed'] = all(checks.values())
    evidence['seconds'] = time.time() - started
    evidence['finished_at'] = now()
    OUT.mkdir(exist_ok=True)
    path = OUT / f'{arm}.json'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(evidence, indent=1) + '\n')
    temporary.replace(path)
    print(json.dumps({'arm': arm, 'passed': evidence['passed'], 'failed': [k for k, v in checks.items() if not v],
                      'seconds': round(evidence['seconds'])}), flush=True)
    if not evidence['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

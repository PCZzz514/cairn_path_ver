"""Foreground GPU3 evidence: original per-pair DSN inter-graph attention (OLD,
read-only cairn_dsn) versus the packed copy (NEW, dsn_gpu3).

Real seed-42 epoch-1 train batches of 128 rows (identical shuffle RNG
consumption), copied parameters, identical RNG for train-mode dropout and PVR
sampling. Discrepancies are max-norm relative: ||new - old||_inf / ||old||_inf
per tensor. Writes evidence/gpu3_evidence.json and .md; no training.
"""
import copy
import json
import math
import os
import platform
import random
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch.utils.data import DataLoader
from torch.utils.data._utils.pin_memory import pin_memory
from torch_geometric.data import Batch
from torch_geometric.utils import add_self_loops, remove_self_loops
import yaml
from cairn_dsn.data import Collator as OldCollator
from cairn_dsn.model import DSNRareDDI as OldModel
from dsn_gpu3.data import Collator, DDIDataset, to_device
from dsn_gpu3.model import DSNRareDDI as NewModel
from dsn_gpu3.status import update_status

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / 'benchmarks/gpu3_speed'
OUT = BENCH / 'evidence'
TRAINED = ROOT / 'outputs/path_single_drugout/best.pt'  # read-only production checkpoint
NUMERIC_BATCHES = 3
WARMUP, MEASURED = 5, 30
# FP32 tolerances on max-norm relative discrepancy (alpha: absolute, alphas lie
# in [0, 1]). FP64 packed-vs-per-pair must agree to double roundoff. Model
# parameter gradients: max_abs <= rel * ||g_old||_inf + atol * G, G = largest
# parameter-gradient magnitude, because some gradients are analytically zero
# (softmax shift invariance) and hold only ~1e-10 roundoff residue.
TOL = {'attention_forward_rel': 1e-5, 'attention_alpha_abs': 1e-6, 'attention_grad_rel': 1e-4,
       'attention_fp64_rel': 1e-10, 'model_forward_rel': 1e-4, 'model_grad_rel': 1e-3,
       'model_grad_atol_of_global_scale': 1e-7}
DEVICE = torch.device('cuda:0')


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_gpu(raw):
    """to_device without mutating the stored pinned batch (PyG Batch.to is in place)."""
    return to_device({k: copy.copy(v) if isinstance(v, Batch) else v for k, v in raw.items()}, DEVICE)


def sync():
    torch.cuda.synchronize(DEVICE)


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


def within(summary, tolerance):
    return summary['worst_rel'] is not None and summary['worst_rel'] <= tolerance


def gradient_check(table, noise):
    """Every parameter-gradient tensor of ONE batch; G is that batch's scale."""
    scale = max(v['ref_max_abs'] for v in table.values())
    atol = TOL['model_grad_atol_of_global_scale'] * scale
    failures = {k: v for k, v in table.items() if v['max_abs'] > atol + TOL['model_grad_rel'] * v['ref_max_abs']}
    below = {k: {'new_vs_old': v, 'old_repeat_vs_old': noise.get(k)} for k, v in table.items() if v['ref_max_abs'] <= atol}
    return {'passed': not failures, 'tensors_checked': len(table), 'global_gradient_scale': scale, 'atol': atol,
            'failures': failures, 'tensors_below_atol_analytically_zero_candidates': below,
            'worst_excluding_below_atol': worst({k: v for k, v in table.items() if k not in below})}


def prefixed(prefix, table):
    return {f'{prefix}.{k}': v for k, v in table.items()}


def stats(values):
    values = np.asarray(values)
    return {'n': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
            'std': float(values.std()), 'min': float(values.min()), 'p90': float(np.percentile(values, 90)),
            'max': float(values.max())}


def contention():
    jobs = run(['ps', '-eo', 'pid,ppid,pcpu,etime,args', '--sort=pid'])
    jobs = [line[:200] for line in jobs.splitlines() if 'cairn_dsn.train' in line]
    return {'time': now(), 'loadavg': os.getloadavg(), 'cpu_count': os.cpu_count(),
            'this_process_affinity_cpus': len(os.sched_getaffinity(0)),
            'gpu3': run(['nvidia-smi', '-i', '3', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
                         '--format=csv,noheader']),
            'gpu3_compute_apps': [line for line in run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                                        '--format=csv,noheader']).splitlines()
                                  if os.environ['GPU3_UUID'] in line],
            'original_cairn_dsn_train_processes': jobs}


def gpu3_clocks():
    return run(['nvidia-smi', '-i', '3', '--query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,pstate,'
                'clocks_event_reasons.active', '--format=csv,noheader'])


def environment(config):
    properties = torch.cuda.get_device_properties(DEVICE)
    uuid = str(properties.uuid)
    expected = os.environ['GPU3_UUID'].removeprefix('GPU-')
    if uuid.removeprefix('GPU-') != expected:
        raise SystemExit(f'Visible CUDA device {uuid} is not physical GPU3 {expected}')
    return {'hostname': socket.gethostname(), 'python': platform.python_version(), 'torch': torch.__version__,
            'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'torch_geometric': torch_geometric.__version__, 'numpy': np.__version__,
            'driver': run(['nvidia-smi', '-i', '3', '--query-gpu=driver_version', '--format=csv,noheader']),
            'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
            'CUDA_DEVICE_ORDER': os.environ.get('CUDA_DEVICE_ORDER'), 'device_count': torch.cuda.device_count(),
            'gpu_name': properties.name, 'gpu_uuid': uuid, 'gpu3_uuid_nvidia_smi': os.environ['GPU3_UUID'],
            'gpu_pci_bus_id': properties.pci_bus_id, 'physical_gpu3_verified': True,
            'torch_threads': torch.get_num_threads(), 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32, 'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cpu_threads_config': config['cpu_threads'], 'num_workers_config': config['num_workers']}


def structure(old_raw, new_raw):
    """Exact per-pair edge lists against PyG's own default GATConv loop rule."""
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
    # What an unmodified default GATConv on this packed graph WOULD do.
    n_loops = min(ga.num_nodes, gb.num_nodes)
    result['naive_default_gat_would_remove_valid_edges'] = result['global_diagonal_non_loop_edges_kept']
    result['naive_default_gat_would_add_cross_pair_loops'] = int((ga.batch[:n_loops] != gb.batch[:n_loops]).sum())
    result['passed'] = (result['direction_h_to_t_exact'] and result['direction_t_to_h_exact'] and
                        result['all_edges_within_own_pair'] and result['local_loops_appended_last'] and
                        result['segments_cover_packed'] and result['nh_lt_nt'] > 0 and result['nh_gt_nt'] > 0)
    return result


def capture_inter_inputs(model, batch, seed):
    """Real 128-dimensional interAtt inputs of both CSE blocks."""
    captured = []
    hooks = [block.interAtt.register_forward_pre_hook(
        lambda module, args: captured.append((args[0].x.detach().clone(), args[1].x.detach().clone())))
        for block in model.cse.blocks]
    with torch.no_grad():
        torch.manual_seed(seed)
        model(batch)
    for hook in hooks:
        hook.remove()
    return captured


def attention_pass(block, hx, tx, edges, ptrs, weights, dtype):
    h = hx.to(dtype).clone().requires_grad_()
    t = tx.to(dtype).clone().requires_grad_()
    if ptrs is None:
        ho, to = block.inter_pair_batch(NS(x=h), NS(x=t), edges)
    else:
        ho, to = block.inter_pair_batch(NS(x=h, ptr=ptrs[0]), NS(x=t, ptr=ptrs[1]), edges)
    loss = (ho * weights[0].to(dtype)).sum() + (to * weights[1].to(dtype)).sum()
    names = [name for name, _ in block.interAtt.named_parameters()]
    grads = torch.autograd.grad(loss, [h, t] + list(block.interAtt.parameters()))
    tensors = {'out.h_rep': ho, 'out.t_rep': to}
    gradients = {'input.h': grads[0], 'input.t': grads[1]}
    gradients.update({f'param.{n}': g for n, g in zip(names, grads[2:])})
    return tensors, gradients


def attention_compare(old_block, new_block, hx, tx, old_edges, packed, ga, gb, seed):
    generator = torch.Generator(DEVICE).manual_seed(seed)
    width = old_block.interAtt.inter.heads * old_block.interAtt.inter.out_channels
    weights = (torch.randn(hx.shape[0], width, device=DEVICE, generator=generator),
               torch.randn(tx.shape[0], width, device=DEVICE, generator=generator))
    ptrs = (ga.ptr, gb.ptr)
    old32 = attention_pass(old_block, hx, tx, old_edges, ptrs, weights, torch.float32)
    new32 = attention_pass(new_block, hx, tx, packed, None, weights, torch.float32)
    old64 = attention_pass(copy.deepcopy(old_block).double(), hx, tx, old_edges, ptrs, weights, torch.float64)
    new64 = attention_pass(copy.deepcopy(new_block).double(), hx, tx, packed, None, weights, torch.float64)

    def compare(a, b):
        return ({k: discrepancy(a[0][k], b[0][k]) for k in a[0]}, {k: discrepancy(a[1][k], b[1][k]) for k in a[1]})

    result = {}
    for label, (a, b) in {'new32_vs_old32': (new32, old32), 'old32_vs_old64': (old32, old64),
                          'new32_vs_old64': (new32, old64), 'new64_vs_old64': (new64, old64)}.items():
        forward, backward = compare(a, b)
        result[label] = {'forward': worst(forward), 'gradients': worst(backward), 'forward_tensors': forward,
                         'gradient_tensors': backward}
    # Attention coefficients and processed edge lists per pair, both directions.
    gat_old, gat_new = old_block.interAtt.inter, new_block.interAtt.inter
    hin, tin = F.elu(hx), F.elu(tx)
    hp, tp = ga.ptr.tolist(), gb.ptr.tolist()
    with torch.no_grad():
        _, (e_ht, a_ht) = gat_new((hin, tin), packed, return_attention_weights=True)
        _, (e_th, a_th) = gat_new((tin, hin), packed[[1, 0]], return_attention_weights=True)
        edges_equal, alpha_abs, seg = True, 0., 0
        for i, local in enumerate(old_edges):
            hs, he, ts, te = hp[i], hp[i + 1], tp[i], tp[i + 1]
            _, (o_ht, b_ht) = gat_old((hin[hs:he], tin[ts:te]), local, return_attention_weights=True)
            _, (o_th, b_th) = gat_old((tin[ts:te], hin[hs:he]), local[[1, 0]], return_attention_weights=True)
            n = o_ht.shape[1]
            offset = torch.tensor([[hs], [ts]], device=DEVICE)
            edges_equal &= (torch.equal(e_ht[:, seg:seg + n] - offset, o_ht) and
                            torch.equal(e_th[:, seg:seg + n] - offset[[1, 0]], o_th))
            alpha_abs = max(alpha_abs, (a_ht[seg:seg + n] - b_ht).abs().max().item(),
                            (a_th[seg:seg + n] - b_th).abs().max().item())
            seg += n
        pair = ga.batch[packed[0]]
        local_loop = (packed[0] - ga.ptr[pair]) == (packed[1] - gb.ptr[pair])
        global_diagonal = (packed[0] == packed[1]) & ~local_loop
    result['attention_coefficients'] = {
        'processed_edge_lists_equal_both_directions': bool(edges_equal) and seg == packed.shape[1],
        'max_abs_alpha_difference': alpha_abs,
        'global_diagonal_non_loop_edges': int(global_diagonal.sum()),
        'global_diagonal_min_alpha_h_to_t': a_ht[global_diagonal].min().item() if global_diagonal.any() else None,
        'global_diagonal_min_alpha_t_to_h': a_th[global_diagonal].min().item() if global_diagonal.any() else None}
    return result


def isolation(new_block, hx, tx, packed, ga, gb):
    """Gradient of one pair's outputs must vanish on every other pair's inputs."""
    h, t = hx.clone().requires_grad_(), tx.clone().requires_grad_()
    ho, to = new_block.inter_pair_batch(NS(x=h), NS(x=t), packed)
    hp, tp = ga.ptr.tolist(), gb.ptr.tolist()
    violations, inside_zero_rows = 0, 0
    for i in range(ga.num_graphs):
        hs, he, ts, te = hp[i], hp[i + 1], tp[i], tp[i + 1]
        gh, gt = torch.autograd.grad(ho[hs:he].sum() + to[ts:te].sum(), [h, t], retain_graph=True)
        outside_h = torch.cat([gh[:hs], gh[he:]])
        outside_t = torch.cat([gt[:ts], gt[te:]])
        violations += int((outside_h != 0).any(1).sum() + (outside_t != 0).any(1).sum())
        inside_zero_rows += int((gh[hs:he] == 0).all(1).sum() + (gt[ts:te] == 0).all(1).sum())
    return {'pairs_checked': ga.num_graphs, 'nonzero_gradient_rows_in_other_pairs': violations,
            'zero_gradient_rows_inside_own_pair': inside_zero_rows,
            'passed': violations == 0 and inside_zero_rows == 0}


def model_pass(model, raw, seed, config, beta):
    batch = to_gpu(raw)
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
        evaluated = model(to_gpu(raw))
    forward['eval.logits'] = evaluated['logits']
    forward['eval.rec'] = evaluated['rec']
    model.zero_grad(set_to_none=True)
    return forward, out['path_mask'].detach(), params, inputs


def model_compare(a, b):
    forward = {k: discrepancy(a[0][k], b[0][k]) for k in a[0]}
    params = {k: discrepancy(a[2][k], b[2][k]) for k in a[2]} if a[2].keys() == b[2].keys() else None
    inputs = {k: discrepancy(a[3][k], b[3][k]) for k in a[3]}
    return {'forward': worst(forward), 'parameter_gradients': worst(params) if params else 'GRAD SET MISMATCH',
            'input_gradients': worst(inputs), 'path_mask_equal': torch.equal(a[1], b[1]),
            'gradient_parameter_count': len(a[2]), 'forward_tensors': forward, 'parameter_tensors': params,
            'input_tensors': inputs}


def timed_block(name, model, raws, mode, config, warmup_updates, fresh, seed):
    """Every block starts from the same parameters, a fresh AdamW and the same seed."""
    model.zero_grad(set_to_none=True)
    model.load_state_dict(fresh, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    update = 0
    torch.cuda.empty_cache()
    sync()
    torch.manual_seed(seed)
    clocks_start, load_start = gpu3_clocks(), os.getloadavg()
    baseline = torch.cuda.memory_allocated(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    phases = {key: [] for key in ['transfer', 'forward', 'backward', 'optimizer', 'fullstep']}
    model.eval() if mode == 'eval' else model.train()
    for k, raw in enumerate(raws):
        sync()
        t0 = time.perf_counter()
        batch = to_gpu(raw)
        if mode == 'eval':
            sync()
            t1 = time.perf_counter()
            with torch.no_grad():
                model(batch)
            sync()
            t4 = time.perf_counter()
            if k >= WARMUP:
                phases['transfer'].append(t1 - t0)
                phases['forward'].append(t4 - t1)
                phases['fullstep'].append(t4 - t0)
            continue
        if mode == 'breakdown':
            sync()
        t1 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        result = model(batch)
        ce = F.cross_entropy(result['logits'], batch['label'])
        beta = config['kl_weight'] * min(1., (update + 1) / warmup_updates)
        loss = ce + config['reconstruction_weight'] * result['rec'] + beta * result['kl']
        if mode == 'breakdown':
            sync()
        t2 = time.perf_counter()
        loss.backward()
        if mode == 'breakdown':
            sync()
        t3 = time.perf_counter()
        optimizer.step()
        sync()
        t4 = time.perf_counter()
        update += 1
        if k >= WARMUP:
            if mode == 'breakdown':
                phases['transfer'].append(t1 - t0)
                phases['forward'].append(t2 - t1)
                phases['backward'].append(t3 - t2)
                phases['optimizer'].append(t4 - t3)
            phases['fullstep'].append(t4 - t0)
    model.zero_grad(set_to_none=True)
    return {'model': name, 'mode': mode, 'seed': seed, 'initial_state': 'fresh seed42 init, fresh AdamW',
            'optimizer_updates': update, 'warmup_steps': WARMUP, 'measured_steps': len(raws) - WARMUP,
            'seconds': {k: stats(v) for k, v in phases.items() if v},
            'per_step_fullstep_seconds': phases['fullstep'], 'gpu3_clocks_start': clocks_start,
            'gpu3_clocks_end': gpu3_clocks(), 'loadavg_start': load_start, 'loadavg_end': os.getloadavg(),
            'memory': {'baseline_allocated_mib': baseline / 2 ** 20,
                       'peak_allocated_mib': torch.cuda.max_memory_allocated(DEVICE) / 2 ** 20,
                       'peak_activation_mib': (torch.cuda.max_memory_allocated(DEVICE) - baseline) / 2 ** 20,
                       'peak_reserved_mib': torch.cuda.max_memory_reserved(DEVICE) / 2 ** 20}}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3':
        raise SystemExit('Evidence requires CUDA_VISIBLE_DEVICES=3 (physical GPU3 only)')
    started = time.time()
    OUT.mkdir(exist_ok=True)
    update_status('evidence_running', evidence={'script': str(Path(__file__)), 'started_at': now()})
    config = yaml.safe_load((BENCH / 'configs/path_single_drugout.yaml').read_text())
    torch.set_num_threads(config['cpu_threads'])
    evidence = {'environment': environment(config), 'contention_start': contention(), 'tolerances': TOL,
                'relative_definition': 'per tensor ||new - old||_inf / ||old||_inf (max-norm relative); '
                                       'worst = maximum over tensors, batches and parameter sets'}
    prepared = ROOT / config['prepared']
    assets = torch.load(prepared / 'assets.pt', map_location='cpu', weights_only=False)

    # Training order: seed 42, construct model, then the epoch-1 shuffle.
    seed_all(config['seed'])
    new_init = NewModel(assets, config).state_dict()
    new_rng = torch.get_rng_state()
    seed_all(config['seed'])
    old_model = OldModel(assets, config)
    old_rng = torch.get_rng_state()
    old_init = old_model.state_dict()
    evidence['initialization'] = {
        'state_dict_names_equal': list(new_init) == list(old_init),
        'state_dict_shapes_equal': all(new_init[k].shape == old_init[k].shape for k in old_init),
        'seed42_init_bitwise_equal': all(torch.equal(new_init[k], old_init[k]) for k in old_init),
        'cpu_rng_after_construction_equal': torch.equal(new_rng, old_rng),
        'state_dict_entries': len(old_init)}
    del new_init
    train = DDIDataset(prepared, config['protocol'], 'train')
    # Same RandomSampler/iterator RNG draws as the training DataLoader (workers
    # do not change the permutation); only dataset positions are taken here.
    order = iter(DataLoader(range(len(train)), batch_size=config['batch_size'], shuffle=True))
    n_batches = WARMUP + MEASURED
    positions = [next(order).tolist() for _ in range(n_batches)]
    old_collate, new_collate = OldCollator(assets, prepared, True), Collator(assets, prepared, True)
    old_raws, new_raws, collate_seconds = [], [], {'old': [], 'new': []}
    for idx in positions:
        rows = [train[i] for i in idx]
        t0 = time.perf_counter()
        old_raw = old_collate(rows)
        t1 = time.perf_counter()
        new_raw = new_collate(rows)
        t2 = time.perf_counter()
        collate_seconds['old'].append(t1 - t0)
        collate_seconds['new'].append(t2 - t1)
        assert torch.equal(old_raw['row_index'], new_raw['row_index']) and torch.equal(old_raw['label'], new_raw['label'])
        old_raws.append(pin_memory(old_raw))
        new_raws.append(pin_memory(new_raw))
    evidence['batches'] = {
        'source': 'train split, seed42 epoch-1 shuffle order, first batches, batch_size 128',
        'train_rows': len(train), 'batch_ids_epoch1': list(range(n_batches)),
        'numerical_batch_ids': list(range(NUMERIC_BATCHES)), 'timing_batch_ids': list(range(n_batches)),
        'row_index_per_batch': [raw['row_index'].tolist() for raw in old_raws],
        'collate_seconds_single_process': {k: stats(v) for k, v in collate_seconds.items()}}
    evidence['structure'] = [structure(o, n) for o, n in zip(old_raws, new_raws)]
    update_status('evidence_running', evidence={'script': str(Path(__file__)), 'started_at': now(),
                                                'progress': 'structure checked; numerical comparison running'})

    old_model = old_model.to(DEVICE)
    new_model = NewModel(assets, config).to(DEVICE)
    fresh = {k: v.clone() for k, v in old_model.state_dict().items()}
    trained = torch.load(TRAINED, map_location='cpu', weights_only=False)
    parameter_sets = {'fresh_seed42_init': fresh, 'trained_production_best': trained['model']}
    evidence['parameter_sets'] = {'fresh_seed42_init': 'seed42 construction (same as five-epoch run)',
                                  'trained_production_best': {'path': str(TRAINED), 'read_only': True,
                                                              'epoch': trained['epoch'], 'update': trained['update']}}
    del trained
    # Original schedule: epochs100 x len(train loader), 20% warmup (23300 for 1165 batches).
    warmup_updates = max(1, math.ceil(config['kl_warmup_fraction'] * config['epochs'] * math.ceil(len(train) / config['batch_size'])))
    beta = config['kl_weight']  # post-warmup schedule value, so KL contributes to gradients
    numerical = {}
    for set_name, state in parameter_sets.items():
        old_model.load_state_dict(state, strict=True)
        new_model.load_state_dict(old_model.state_dict(), strict=True)
        attention, gradient_checks = [], []
        model_new = {'forward': {}, 'parameter': {}, 'input': {}}
        model_noise = {'forward': {}, 'parameter': {}, 'input': {}}
        records = {'new_vs_old': [], 'old_repeat_vs_old': []}
        isolation_result = None
        for b in range(NUMERIC_BATCHES):
            seed = 1000 + b
            captured = capture_inter_inputs(new_model, to_gpu(new_raws[b]), seed)
            old_edges = to_device(old_raws[b]['cross_edges'], DEVICE)
            packed = new_raws[b]['cross_edges'].to(DEVICE)
            ga, gb = copy.copy(new_raws[b]['graph_a']).to(DEVICE), copy.copy(new_raws[b]['graph_b']).to(DEVICE)
            for k, (hx, tx) in enumerate(captured):
                result = attention_compare(old_model.cse.blocks[k], new_model.cse.blocks[k], hx, tx,
                                           old_edges, packed, ga, gb, seed)
                result.update({'batch_id': b, 'cse_block': k})
                attention.append(result)
            if b == 0:
                isolation_result = isolation(new_model.cse.blocks[0], *captured[0], packed, ga, gb)
            old = model_pass(old_model, old_raws[b], seed, config, beta)
            old_repeat = model_pass(old_model, old_raws[b], seed, config, beta)
            new = model_pass(new_model, new_raws[b], seed, config, beta)
            versus = model_compare(new, old)
            noise = model_compare(old_repeat, old)
            for label, value, other in [('new_vs_old', versus, new), ('old_repeat_vs_old', noise, old_repeat)]:
                records[label].append({'batch_id': b, 'forward': value['forward'],
                                       'parameter_gradients': value['parameter_gradients'],
                                       'input_gradients': value['input_gradients'],
                                       'path_mask_equal': value['path_mask_equal'],
                                       'gradient_parameter_count': value['gradient_parameter_count'],
                                       'loss_old': old[0]['loss'].item(), 'loss_compared': other[0]['loss'].item()})
            # Every tensor of every batch is kept; worst abs/rel aggregate independently.
            for total, value in [(model_new, versus), (model_noise, noise)]:
                total['forward'].update(prefixed(f'batch{b}', value['forward_tensors']))
                total['parameter'].update(prefixed(f'batch{b}', value['parameter_tensors'] or {}))
                total['input'].update(prefixed(f'batch{b}', value['input_tensors']))
            if versus['parameter_tensors'] is not None:
                check = gradient_check(versus['parameter_tensors'], noise['parameter_tensors'] or {})
                gradient_checks.append({'batch_id': b, **check})
            else:
                gradient_checks.append({'batch_id': b, 'passed': False, 'failures': 'gradient parameter sets differ'})
            update_status('evidence_running', evidence={'script': str(Path(__file__)), 'started_at': now(),
                                                        'progress': f'numerical {set_name} batch {b} done'})
        attention_worst = {label: {'forward': worst({f'batch{r["batch_id"]}.block{r["cse_block"]}.{k}': v
                                                     for r in attention for k, v in r[label]['forward_tensors'].items()}),
                                   'gradients': worst({f'batch{r["batch_id"]}.block{r["cse_block"]}.{k}': v
                                                       for r in attention for k, v in r[label]['gradient_tensors'].items()})}
                           for label in ['new32_vs_old32', 'old32_vs_old64', 'new32_vs_old64', 'new64_vs_old64']}
        coefficients = [r['attention_coefficients'] for r in attention]
        for r in attention:
            for label in ['new32_vs_old32', 'old32_vs_old64', 'new32_vs_old64', 'new64_vs_old64']:
                r[label] = {'forward': r[label]['forward'], 'gradients': r[label]['gradients']}
        below_atol = {f'batch{c["batch_id"]}.{k}' for c in gradient_checks
                      for k in c.get('tensors_below_atol_analytically_zero_candidates', {})}
        numerical[set_name] = {
            'attention_per_batch_block': attention, 'attention_worst': attention_worst,
            'attention_coefficients_worst': {
                'processed_edge_lists_equal_both_directions': all(c['processed_edge_lists_equal_both_directions'] for c in coefficients),
                'max_abs_alpha_difference': max(c['max_abs_alpha_difference'] for c in coefficients),
                'global_diagonal_non_loop_edges': sum(c['global_diagonal_non_loop_edges'] for c in coefficients),
                'global_diagonal_min_alpha': min(min(c['global_diagonal_min_alpha_h_to_t'], c['global_diagonal_min_alpha_t_to_h'])
                                                 for c in coefficients if c['global_diagonal_non_loop_edges'])},
            'cross_pair_isolation_batch0_block0': isolation_result,
            'model_per_batch': records,
            'model_worst_new_vs_old': {k: worst(v) for k, v in model_new.items()},
            'model_worst_old_repeat_vs_old_noise_floor': {k: worst(v) for k, v in model_noise.items()},
            'model_parameter_gradient_tensors_new_vs_old': model_new['parameter'],
            'model_parameter_gradient_tensors_noise_floor': model_noise['parameter'],
            'model_parameter_gradient_check': {
                'passed': all(c['passed'] for c in gradient_checks), 'per_batch': gradient_checks,
                'tensors_checked': sum(c.get('tensors_checked', 0) for c in gradient_checks),
                'worst_excluding_below_atol': worst({k: v for k, v in model_new['parameter'].items()
                                                     if k not in below_atol})}}
    evidence['numerical'] = numerical
    del parameter_sets

    # Timing: identical fresh parameters and real batches; ABBA blocks.
    update_status('evidence_running', evidence={'script': str(Path(__file__)), 'started_at': now(),
                                                'progress': 'numerical done; timing running'})
    models = {'old': old_model, 'new': new_model}
    raws = {'old': old_raws, 'new': new_raws}
    evidence['contention_before_timing'] = contention()
    blocks = []
    for name, mode in [('old', 'breakdown'), ('new', 'breakdown'), ('new', 'fullstep'), ('old', 'fullstep'),
                       ('old', 'eval'), ('new', 'eval'), ('new', 'eval'), ('old', 'eval')]:
        blocks.append(timed_block(name, models[name], raws[name], mode, config, warmup_updates, fresh, config['seed']))
        update_status('evidence_running', evidence={'script': str(Path(__file__)), 'started_at': now(),
                                                    'progress': f'timing block {len(blocks)}/8 done'})
    evidence['contention_after_timing'] = contention()
    summary = {}
    for mode, phases in [('breakdown', ['transfer', 'forward', 'backward', 'optimizer', 'fullstep']),
                         ('fullstep', ['fullstep']), ('eval', ['transfer', 'forward', 'fullstep'])]:
        for phase in phases:
            medians = {name: [b['seconds'][phase]['median'] for b in blocks if b['model'] == name and b['mode'] == mode]
                       for name in ['old', 'new']}
            old_s, new_s = float(np.mean(medians['old'])), float(np.mean(medians['new']))
            summary[f'{mode}.{phase}'] = {'old_median_seconds': old_s, 'new_median_seconds': new_s,
                                          'old_over_new': old_s / new_s}
    evidence['timing'] = {
        'setup': f'GPU3 only; same {n_batches} real batches in the same row order; EVERY block restarts from the '
                 f'identical fresh seed42 parameters, a fresh AdamW and torch.manual_seed({config["seed"]}); '
                 f'{WARMUP} warmup + {MEASURED} measured steps per block; ABBA block order; CUDA synchronized '
                 'around each timed region; transfer = pinned host batch to GPU (non_blocking, as training); '
                 'fullstep = transfer + zero_grad + forward + loss + backward + AdamW step; breakdown blocks '
                 'synchronize between phases, fullstep blocks only at step boundaries; eval = no_grad eval forward. '
                 'Collation (CPU, DataLoader workers in training) is reported separately.',
        'blocks': blocks, 'median_of_block_medians': summary,
        'note': 'Measured per-step times on these batches only; four original jobs (GPU4-7) share host CPUs.'}

    checks = {'gpu3_verified': evidence['environment']['physical_gpu3_verified'],
              'initialization_identical': all(v for k, v in evidence['initialization'].items() if k != 'state_dict_entries'),
              'structure_all_batches': all(s['passed'] for s in evidence['structure']),
              'global_diagonal_edges_present_and_kept': sum(s['global_diagonal_non_loop_edges_kept'] for s in evidence['structure']) > 0}
    for set_name, value in numerical.items():
        aw = value['attention_worst']
        checks[f'{set_name}.attention_forward'] = within(aw['new32_vs_old32']['forward'], TOL['attention_forward_rel'])
        checks[f'{set_name}.attention_gradients'] = within(aw['new32_vs_old32']['gradients'], TOL['attention_grad_rel'])
        checks[f'{set_name}.attention_fp64_equivalence'] = (within(aw['new64_vs_old64']['forward'], TOL['attention_fp64_rel']) and
                                                             within(aw['new64_vs_old64']['gradients'], TOL['attention_fp64_rel']))
        coefficient = value['attention_coefficients_worst']
        checks[f'{set_name}.attention_edges_and_alpha'] = (coefficient['processed_edge_lists_equal_both_directions'] and
                                                            coefficient['max_abs_alpha_difference'] <= TOL['attention_alpha_abs'] and
                                                            coefficient['global_diagonal_min_alpha'] > 0)
        checks[f'{set_name}.cross_pair_isolation'] = value['cross_pair_isolation_batch0_block0']['passed']
        mw = value['model_worst_new_vs_old']
        checks[f'{set_name}.model_forward'] = within(mw['forward'], TOL['model_forward_rel'])
        checks[f'{set_name}.model_parameter_gradients'] = value['model_parameter_gradient_check']['passed']
        checks[f'{set_name}.model_input_gradients'] = within(mw['input'], TOL['model_grad_rel'])
        checks[f'{set_name}.path_mask_and_grad_sets'] = all(
            r['path_mask_equal'] and isinstance(r['parameter_gradients'], dict) for r in value['model_per_batch']['new_vs_old'])
    evidence['checks'] = checks
    evidence['all_numerical_checks_passed'] = all(checks.values())
    evidence['timing_measured'] = True
    evidence['seconds'] = time.time() - started
    evidence['finished_at'] = now()
    path = OUT / 'gpu3_evidence.json'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(evidence, indent=1) + '\n')
    temporary.replace(path)
    (OUT / 'gpu3_evidence.md').write_text(markdown(evidence))
    update_status('evidence_complete_awaiting_codex_review',
                  evidence={'json': str(path), 'markdown': str(OUT / 'gpu3_evidence.md'), 'finished_at': now(),
                            'all_numerical_checks_passed': evidence['all_numerical_checks_passed'],
                            'failed_checks': [k for k, v in checks.items() if not v],
                            'timing_measured': True, 'speed': {k: v['old_over_new'] for k, v in summary.items()}})
    print(json.dumps({'all_numerical_checks_passed': evidence['all_numerical_checks_passed'],
                      'failed_checks': [k for k, v in checks.items() if not v], 'timing': summary}, indent=1))
    if not evidence['all_numerical_checks_passed']:
        raise SystemExit('numerical checks failed; see gpu3_evidence.json')


def fmt(value):
    return 'n/a' if value is None else f'{value:.3e}'


def markdown(e):
    lines = ['# GPU3 OLD (per-pair) vs NEW (packed) DSN inter-graph attention evidence', '',
             f"Finished {e['finished_at']}; {e['seconds']:.0f} s. GPU: {e['environment']['gpu_name']} "
             f"{e['environment']['gpu_uuid']} (nvidia-smi GPU3 {e['environment']['gpu3_uuid_nvidia_smi']}), "
             f"torch {e['environment']['torch']} CUDA {e['environment']['torch_cuda']}, PyG {e['environment']['torch_geometric']}, "
             f"driver {e['environment']['driver']}, TF32 matmul {e['environment']['tf32_matmul']}.", '',
             f"Relative discrepancy: {e['relative_definition']}.", '',
             f"**All numerical checks passed: {e['all_numerical_checks_passed']}**", '', '## Checks', '']
    lines += [f'- {k}: {v}' for k, v in e['checks'].items()]
    s = e['structure']
    lines += ['', '## Structure (all timing batches)', '',
              f"- batches {len(s)}, pairs nh<nt {sum(x['nh_lt_nt'] for x in s)}, nh>nt {sum(x['nh_gt_nt'] for x in s)}, "
              f"nh=nt {sum(x['nh_eq_nt'] for x in s)}; local loops {sum(x['local_loops'] for x in s)}",
              f"- global row==col non-loop edges kept: {sum(x['global_diagonal_non_loop_edges_kept'] for x in s)}; "
              f"a default-loop GAT on the packed graph would wrongly add {sum(x['naive_default_gat_would_add_cross_pair_loops'] for x in s)} cross-pair loops",
              '', '## Numerical worst discrepancies', '',
              '| parameters | comparison | quantity | worst rel | tensor | worst abs | tensor |', '|---|---|---|---|---|---|---|']
    for set_name, value in e['numerical'].items():
        for label, parts in value['attention_worst'].items():
            for quantity, w in parts.items():
                lines.append(f"| {set_name} | attention {label} | {quantity} | {fmt(w['worst_rel'])} | {w['worst_rel_tensor']} | "
                             f"{fmt(w['worst_abs'])} | {w['worst_abs_tensor']} |")
        for label, key in [('model NEW vs OLD', 'model_worst_new_vs_old'), ('model OLD repeat vs OLD', 'model_worst_old_repeat_vs_old_noise_floor')]:
            for quantity, w in value[key].items():
                lines.append(f"| {set_name} | {label} | {quantity} | {fmt(w['worst_rel'])} | {w['worst_rel_tensor']} | "
                             f"{fmt(w['worst_abs'])} | {w['worst_abs_tensor']} |")
        g = value['model_parameter_gradient_check']
        r = g['worst_excluding_below_atol']
        lines.append(f"| {set_name} | model NEW vs OLD | parameter grads above atol ({g['tensors_checked']} tensor-batches) | "
                     f"{fmt(r['worst_rel'])} | {r['worst_rel_tensor']} | {fmt(r['worst_abs'])} | {r['worst_abs_tensor']} |")
        for c in g['per_batch']:
            lines.append(f"| {set_name} | gradient gate batch {c['batch_id']} | passed {c['passed']} | | atol {fmt(c['atol'])} | "
                         f"G {fmt(c['global_gradient_scale'])} | failures {len(c['failures'])} |")
            for k, v in c['tensors_below_atol_analytically_zero_candidates'].items():
                lines.append(f"| {set_name} | below-atol grad batch {c['batch_id']} | {k} | new-old abs {fmt(v['new_vs_old']['max_abs'])} | "
                             f"ref {fmt(v['new_vs_old']['ref_max_abs'])} | OLD repeat abs {fmt(v['old_repeat_vs_old']['max_abs'])} | |")
        c = value['attention_coefficients_worst']
        lines.append(f"| {set_name} | alpha per pair | max abs | | | {fmt(c['max_abs_alpha_difference'])} | "
                     f"global-diag edges {c['global_diagonal_non_loop_edges']}, min alpha {fmt(c['global_diagonal_min_alpha'])} |")
    lines += ['', '## Timing (median of block medians, seconds)', '', e['timing']['setup'], '',
              '| measurement | OLD | NEW | OLD/NEW |', '|---|---|---|---|']
    for k, v in e['timing']['median_of_block_medians'].items():
        lines.append(f"| {k} | {v['old_median_seconds']:.4f} | {v['new_median_seconds']:.4f} | {v['old_over_new']:.2f} |")
    lines += ['', '| block (ABBA order) | mode | fullstep median | min | p90 | max | peak activation MiB | peak reserved MiB |',
              '|---|---|---|---|---|---|---|---|']
    for b in e['timing']['blocks']:
        f = b['seconds']['fullstep']
        lines.append(f"| {b['model']} | {b['mode']} | {f['median']:.4f} | {f['min']:.4f} | {f['p90']:.4f} | {f['max']:.4f} | "
                     f"{b['memory']['peak_activation_mib']:.0f} | {b['memory']['peak_reserved_mib']:.0f} |")
    c = e['batches']['collate_seconds_single_process']
    lines += ['', f"Collate (single process, CPU) mean seconds: OLD {c['old']['mean']:.4f}, NEW {c['new']['mean']:.4f}.",
              '', '## Contention', '', f"- before timing: {json.dumps(e['contention_before_timing'])}",
              f"- after timing: {json.dumps(e['contention_after_timing'])}", '']
    return '\n'.join(lines)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        update_status('evidence_failed', evidence={'failed_at': now(), 'error': repr(error)[:2000]})
        raise

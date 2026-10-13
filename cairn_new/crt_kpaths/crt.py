"""Decoupled classifier re-training (cRT) on a frozen parent DSN-DDI model.

Kang et al. ICLR 2020 (1910.09217v2) Sec. 4; official facebookresearch/
classifier-balancing f162adef63fdf592d0b8a0f6e52bc481fc08785f (cls_crt.yaml,
run_networks.py, DotProductClassifier.py, ClassAwareSampler.py).

1. extract: strict-load the parent best.pt, eval mode, all parameters frozen.
   A forward_pre_hook on the last Linear(256,85) captures the real g_ab of the
   unmodified forward (classifier on deterministic mu, dropout off). Every
   batch checks old_last_linear(g) == parent logits; the test softmax is
   compared with the parent's exported test probabilities row by row.
2. heads: a NEW nn.Linear(256,85) (default init, seed-shared between arms) on
   the cached features. cRT = class-aware sampling (4 per class), natural =
   instance shuffling; both floor(Ntrain/512) SGD steps/epoch, batch 512, CE,
   momentum .9, wd 5e-4, CosineAnnealingLR(T_max=10, eta_min=0) stepped after
   each epoch's updates (modern PyTorch order: epoch e trains at
   lr*(1+cos(pi*(e-1)/10))/2, every epoch updates). run_networks.train steps
   it at epoch start, whose effect depends on the PyTorch version; adapted, see
   README. Strict fixed85 validation macro-F1 selection, no early stop.
3. step1: common lr from [.01,.05,.2] at head seed 42 by the mean of the two
   arms' best validation F1 (ties: candidate order); then seeds 43/44 (head
   randomness only; one end-to-end parent). step3 reuses the protocol's step1 lr.
4. test is read only after every selection is final.
"""
import argparse
import copy
from itertools import islice
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from accelerated_four.dsn.data import DDIDataset, Collator, to_device
from accelerated_four.dsn.metrics import classification_metrics, validation_selection_metric
from accelerated_four.dsn.model import DSNRareDDI
from .sampler import ClassAwareSampler

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'crt_kpaths/outputs'
ARMS = ('natural', 'crt')


def resolve(value):
    path = Path(value)
    path = (ROOT / path if not path.is_absolute() else path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Path must remain inside {ROOT}: {path}')
    return path


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_torch(path, value):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def log(**fields):
    print(json.dumps({'time': time.time(), **fields}, allow_nan=False), flush=True)


def load_parent(parent_output, device):
    checkpoint_path = parent_output / 'best.pt'
    parent = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = parent['config']
    prepared = Path(config['prepared'])
    # Step2 parents record their K-Paths topology cache; the original
    # accelerated_four parents read data/prepared.
    path_cache = Path(config.get('path_cache', config['prepared']))
    assets = torch.load(prepared / 'assets.pt', map_location='cpu', weights_only=False)
    model = DSNRareDDI(assets, config)
    model.load_state_dict(parent['model'], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    last = model.classifier[4]
    assert isinstance(last, nn.Linear) and (last.in_features, last.out_features) == (256, 85)
    info = {'checkpoint': str(checkpoint_path), 'epoch': parent['epoch'], 'update': parent['update'],
            'best_metric': parent['best_metric'], 'config': config, 'prepared': str(prepared),
            'path_cache': str(path_cache)}
    return model, assets, info


@torch.no_grad()
def extract(model, assets, info, protocol, split, device, batch_size, limit=None):
    prepared, path_cache = Path(info['prepared']), Path(info['path_cache'])
    dataset = DDIDataset(prepared, protocol, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                        collate_fn=Collator(assets, path_cache, True), pin_memory=True)
    last = model.classifier[4]
    captured = []
    handle = last.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
    features, logits, rows, labels, max_error, started = [], [], [], [], 0., time.time()
    try:
        for step, raw in enumerate(islice(loader, limit)):
            captured.clear()
            output = model(to_device(raw, device))
            assert len(captured) == 1
            g, parent_logits = captured[0], output['logits']
            recomputed = last(g)
            max_error = max(max_error, float((recomputed - parent_logits).abs().max()))
            assert torch.allclose(recomputed, parent_logits, atol=1e-6, rtol=1e-5)
            assert torch.equal(recomputed.argmax(1), parent_logits.argmax(1))
            features.append(g.float().cpu())
            logits.append(parent_logits.float().cpu())
            rows.append(raw['row_index'])
            labels.append(raw['label'])
            if step % 100 == 0:
                log(phase='extract_progress', split=split, step=step, batches=len(loader),
                    seconds=time.time() - started)
    finally:
        handle.remove()
    rows = torch.cat(rows).numpy()
    n = len(rows)
    assert np.array_equal(rows, dataset.rows[:n])
    return {'g': torch.cat(features), 'logits': torch.cat(logits), 'labels': torch.cat(labels),
            'frozen_rows': torch.from_numpy(rows), 'drug_a': torch.from_numpy(dataset.a[:n]),
            'drug_b': torch.from_numpy(dataset.b[:n]), 'pair_ids': torch.from_numpy(dataset.pair_ids[:n]),
            'last_linear_max_abs_error': max_error, 'seconds': time.time() - started}


def parent_test_check(parent_output, test):
    with np.load(parent_output / 'test_probabilities.npz') as data:
        n = len(test['frozen_rows'])
        rows, probabilities = data['frozen_row_indices'][:n], data['probabilities'][:n]
        labels = data['labels'][:n]
    ours = test['logits'].softmax(-1).numpy()
    error = float(np.abs(ours - probabilities).max())
    result = {'rows_checked': n, 'rows_equal': bool(np.array_equal(rows, test['frozen_rows'].numpy())),
              'labels_equal': bool(np.array_equal(labels, test['labels'].numpy())),
              'max_abs_probability_error': error, 'argmax_equal': int((ours.argmax(1) == probabilities.argmax(1)).sum())}
    # Every original test top-1 must be reproduced, not only close probabilities.
    assert result['rows_equal'] and result['labels_equal'] and error <= 1e-5 and result['argmax_equal'] == n, result
    return result


def initial_head(path, seed):
    """One default-initialized nn.Linear(256,85) per head seed (official
    DotProduct_Classifier, stage1_weights false), shared by both arms/lrs."""
    if not path.exists():
        torch.manual_seed(seed)
        save_torch(path, nn.Linear(256, 85).state_dict())
    return torch.load(path, map_location='cpu', weights_only=True)


def train_head(features, arm, lr, seed, init, config, device):
    g, y = features['train']['g'], features['train']['labels']
    g_val, y_val = features['val']['g'], features['val']['labels'].numpy()
    batch = config['batch_size']
    steps = len(y) // batch
    head = nn.Linear(256, 85).to(device)
    head.load_state_dict(init)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=config['momentum'],
                                weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=0.)
    if arm == 'crt':
        sampler = ClassAwareSampler(y.tolist(), config['num_samples_cls'], seed)
    else:
        generator = torch.Generator().manual_seed(seed)
    best, history = {'val_macro_f1_85': -1.}, []
    for epoch in range(1, config['epochs'] + 1):
        started = time.time()
        epoch_lr = optimizer.param_groups[0]['lr']
        if arm == 'crt':
            # A fresh generator over the persistent cycle iterators, as the
            # DataLoader re-iterates the sampler each epoch; epoch_steps batches.
            order = torch.tensor(list(islice(iter(sampler), steps * batch)))
        else:
            order = torch.randperm(len(y), generator=generator)[:steps * batch]
        order = order.view(steps, batch)
        head.train()
        loss_sum, correct = 0., 0
        for step in range(steps):
            index = order[step].to(device)
            logits = head(g[index])
            loss = F.cross_entropy(logits, y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * batch
            correct += int((logits.argmax(1) == y[index]).sum())
        head.eval()
        with torch.no_grad():
            probabilities = head(g_val).softmax(-1).cpu().numpy()
        f1 = validation_selection_metric(y_val, probabilities)
        sampled = np.bincount(y[order.flatten().to(device)].cpu().numpy(), minlength=85)
        record = {'epoch': epoch, 'epoch_lr': epoch_lr, 'steps': steps,
                  'train_ce': loss_sum / (steps * batch), 'train_accuracy': correct / (steps * batch),
                  'val_macro_f1_85': f1, 'val_accuracy': float((probabilities.argmax(1) == y_val).mean()),
                  'sampled_classes': int((sampled > 0).sum()), 'sampled_max_class_rows': int(sampled.max()),
                  'improved': f1 > best['val_macro_f1_85'], 'seconds': time.time() - started}
        history.append(record)
        if record['improved']:
            best = {'val_macro_f1_85': f1, 'epoch': epoch, 'state': copy.deepcopy(head.state_dict())}
        log(phase='head_epoch', arm=arm, base_lr=lr, seed=seed, **record)
        scheduler.step()
    return best, history


def head_name(arm, lr, seed):
    return f'{arm}_lr{lr:g}_seed{seed}'


def run_head(features, output, arm, lr, seed, config, device):
    directory = output / 'heads'
    result_path = directory / f'{head_name(arm, lr, seed)}.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    init = initial_head(directory / f'init_seed{seed}.pt', seed)
    # An interrupted head is re-run from its saved initialization; all its
    # randomness comes from the head seed, so the rerun is the same schedule.
    best, history = train_head(features, arm, lr, seed, init, config, device)
    save_torch(directory / f'{head_name(arm, lr, seed)}_best.pt',
               {'state': best['state'], 'epoch': best['epoch'], 'arm': arm, 'lr': lr, 'seed': seed})
    result = {'arm': arm, 'lr': lr, 'seed': seed, 'best_epoch': best['epoch'],
              'best_val_macro_f1_85': best['val_macro_f1_85'], 'init': f'init_seed{seed}.pt', 'history': history}
    save_json(result_path, result)
    return result


def evaluation(y, probabilities, train_classes):
    metrics, per_class, confusion = classification_metrics(y, probabilities)
    top5 = np.argsort(-probabilities, axis=1)[:, :5]
    metrics['top5_accuracy'] = float((top5 == y[:, None]).any(1).mean())
    # Diagnostic only; the primary metric is always fixed85 macro-F1.
    both = sorted(set(train_classes) & set(np.unique(y).tolist()))
    metrics['diagnostic_train_and_test_positive_classes'] = {
        'class_ids': both, 'count': len(both),
        'macro_f1': float(per_class.set_index('event_id').loc[both, 'f1'].mean())}
    return metrics, per_class, confusion


def class_counts(per_class, labels):
    """85 rows: train/val/test support, test predictions and test P/R/F1."""
    table = pd.DataFrame({'event_id': np.arange(85)})
    for split, y in labels.items():
        table[f'{split}_support'] = np.bincount(y, minlength=85)
    test = per_class[['event_id', 'predicted', 'precision', 'recall', 'f1']]
    return table.merge(test.rename(columns=lambda c: c if c == 'event_id' else f'test_{c}'), on='event_id')


def step2_test_additions(output_dir, labels):
    """Step2 additions from its ALREADY exported test files (after selection).

    Returns test_metrics.json plus top5_accuracy and the train-and-test-positive
    diagnostic (crt.evaluation convention; existing fields are not changed) and
    the 85-row class-count table. Never used for checkpoint selection.
    """
    metrics = json.loads((output_dir / 'test_metrics.json').read_text())
    with np.load(output_dir / 'test_probabilities.npz') as exported:
        y, probabilities = exported['labels'], exported['probabilities']
    assert np.array_equal(y, labels['test'])
    computed, per_class, _ = evaluation(y, probabilities, np.unique(labels['train']).tolist())
    added = {key: computed[key] for key in ['top5_accuracy', 'diagnostic_train_and_test_positive_classes']}
    assert not set(added) & set(metrics)
    return {**metrics, **added}, class_counts(per_class, labels)


def export_head(directory, name, test, probabilities, train_classes, events, extra, labels):
    directory.mkdir(parents=True, exist_ok=True)
    y = test['labels'].numpy()
    metrics, per_class, confusion = evaluation(y, probabilities, train_classes)
    metrics.update(extra)
    np.savez_compressed(directory / f'{name}_test_probabilities.npz', probabilities=probabilities,
                        event_ids=np.arange(85), labels=y, frozen_row_indices=test['frozen_rows'].numpy(),
                        drug_a=test['drug_a'].numpy(), drug_b=test['drug_b'].numpy())
    per_class.merge(events, on='event_id', how='left').to_csv(directory / f'{name}_test_per_class.csv', index=False)
    class_counts(per_class, labels).to_csv(directory / f'{name}_test_class_counts.csv', index=False)
    pd.DataFrame(confusion, index=np.arange(85), columns=np.arange(85)).rename_axis(
        'true_event_id').to_csv(directory / f'{name}_test_confusion.csv')
    save_json(directory / f'{name}_test_metrics.json', metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--measure-batches', type=int,
                        help='Real-data extraction check on the first N test batches only; writes evidence, no features')
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(config['gpu']):
        raise ValueError(f'Requires CUDA_VISIBLE_DEVICES={config["gpu"]}')
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device('cuda:0')
    output, parent_output = resolve(config['output']), resolve(config['parent_output'])
    if not output.is_relative_to(OUTPUTS):
        raise ValueError(f'Output must remain inside {OUTPUTS}')
    (output / 'heads').mkdir(parents=True, exist_ok=True)
    protocol = config['protocol']
    log(phase='start', step=config['step'], protocol=protocol, pid=os.getpid(), output=str(output),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if args.measure_batches:
        model, assets, info = load_parent(parent_output, device)
        test = extract(model, assets, info, protocol, 'test', device, config['extract_batch_size'], args.measure_batches)
        evidence = {'parent': {k: v for k, v in info.items() if k != 'config'}, 'batches': args.measure_batches,
                    'rows': len(test['frozen_rows']), 'seconds': test['seconds'],
                    'last_linear_max_abs_error': test['last_linear_max_abs_error'],
                    'parent_test_check': parent_test_check(parent_output, test)}
        path = ROOT / f'crt_kpaths/evidence/extraction_check_{config["step"]}_{protocol}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, evidence)
        log(phase='measure_complete', **evidence)
        return
    if (output / 'test_complete.json').exists():
        log(phase='already_complete')
        return
    features_path = output / 'features.pt'
    if not features_path.exists():
        model, assets, info = load_parent(parent_output, device)
        features = {split: extract(model, assets, info, protocol, split, device, config['extract_batch_size'])
                    for split in ['train', 'val', 'test']}
        check = parent_test_check(parent_output, features['test'])
        features['parent'] = info
        features['checks'] = {'parent_test': check, **{f'{split}_last_linear_max_abs_error':
                              features[split]['last_linear_max_abs_error'] for split in ['train', 'val', 'test']},
                              'parent_frozen': all(not p.requires_grad for p in model.parameters()),
                              'parent_training_mode': model.training}
        save_torch(features_path, features)
        save_json(output / 'extraction.json', {'parent': info,
                                               'checks': features['checks'],
                                               'rows': {s: len(features[s]['labels']) for s in ['train', 'val', 'test']},
                                               'seconds': {s: features[s]['seconds'] for s in ['train', 'val', 'test']}})
        del model
        torch.cuda.empty_cache()
    features = torch.load(features_path, map_location='cpu', weights_only=False)
    log(phase='features_ready', checks=features['checks'])
    train_split = {'g': features['train']['g'].to(device), 'labels': features['train']['labels'].to(device)}
    data = {'train': train_split, 'val': {'g': features['val']['g'].to(device), 'labels': features['val']['labels']}}
    train_classes = np.unique(features['train']['labels'].numpy()).tolist()
    head_config = config['head']
    # Learning rate: step1 selects on validation only; step3 reuses step1's.
    selection_path = output / 'lr_selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
    elif config['step'] == 'step1':
        dev_seed, candidates = head_config['dev_seed'], head_config['lr_candidates']
        scores = []
        for lr in candidates:
            results = {arm: run_head(data, output, arm, lr, dev_seed, head_config, device) for arm in ARMS}
            scores.append({'lr': lr, **{f'{arm}_best_val_macro_f1_85': results[arm]['best_val_macro_f1_85'] for arm in ARMS},
                           'mean_best_val_macro_f1_85': float(np.mean([results[arm]['best_val_macro_f1_85'] for arm in ARMS]))})
        # max() keeps the FIRST maximum: ties resolve in candidate order.
        chosen = max(scores, key=lambda s: s['mean_best_val_macro_f1_85'])
        selection = {'lr': chosen['lr'], 'rule': 'max mean of natural/cRT best validation fixed85 macro-F1 at dev seed; ties by candidate order',
                     'dev_seed': dev_seed, 'candidates': candidates, 'scores': scores, 'test_used': False}
        save_json(selection_path, selection)
    else:
        source = resolve(config['lr_from'])
        selection = {**json.loads(source.read_text()), 'reused_from': str(source), 'reselected': False}
        save_json(selection_path, selection)
    lr = selection['lr']
    log(phase='lr_selected', lr=lr)
    results = {(arm, seed): run_head(data, output, arm, lr, seed, head_config, device)
               for seed in head_config['seeds'] for arm in ARMS}
    # All selection is final. Test features are used from here on only.
    test = features['test']
    test_g = test['g'].to(device)
    events = pd.read_csv(ROOT / 'data/prepared/ddi_events.csv')
    exports = output / 'test'
    summary = {'step': config['step'], 'protocol': protocol, 'lr': lr, 'seeds': head_config['seeds'],
               'seed_meaning': 'head initialization/sampling repeats on ONE fixed end-to-end parent',
               'parent': {k: v for k, v in features['parent'].items() if k != 'config'}, 'heads': {}}
    parent_probabilities = test['logits'].softmax(-1).numpy()
    labels = {split: features[split]['labels'].numpy() for split in ['train', 'val', 'test']}
    summary['parent_original_head'] = export_head(exports, 'parent_original_head', test, parent_probabilities,
                                                  train_classes, events, {'head': 'parent original classifier'}, labels)
    for (arm, seed), result in results.items():
        name = head_name(arm, lr, seed)
        state = torch.load(output / 'heads' / f'{name}_best.pt', map_location='cpu', weights_only=True)
        head = nn.Linear(256, 85).to(device)
        head.load_state_dict(state['state'])
        head.eval()
        with torch.no_grad():
            probabilities = head(test_g).softmax(-1).cpu().numpy()
        summary['heads'][name] = export_head(exports, name, test, probabilities, train_classes, events, {
            'arm': arm, 'lr': lr, 'seed': seed, 'selected_epoch': result['best_epoch'],
            'selected_val_macro_f1_85': result['best_val_macro_f1_85'], 'selection': 'validation only'}, labels)
    keys = ['macro_f1_85', 'accuracy', 'top5_accuracy', 'macro_roc_auc_supported_classes']
    summary['aggregate'] = {arm: {key: {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)), 'values': values}
                                  for key in keys for values in [[summary['heads'][head_name(arm, lr, s)][key]
                                                                  for s in head_config['seeds']]]}
                            for arm in ARMS}
    save_json(output / 'summary.json', summary)
    save_json(output / 'test_complete.json', {'lr': lr, 'heads': sorted(summary['heads']), 'completed_at': time.time()})
    log(phase='test_complete', aggregate=summary['aggregate'])


if __name__ == '__main__':
    main()

"""Full DDI training, fixed85 validation selection, selected-checkpoint test export.

rareddie_control copy of the accelerated_four trainer: RareDDIE-only model
(no DSN inter-graph attention, no path), molecule-only collator, outputs only
under rareddie_control/outputs. --fresh is the initial launch only; --resume
and --test-only continue this run's own checkpoints with the exact config.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from accelerated_four.dsn.metrics import classification_metrics, validation_selection_metric
from .data import DDIDataset, Collator, to_device
from .model import GROUPS, RareDDIEControl, parameter_inventory

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'rareddie_control/outputs'
GPU = {'single_drugout': 6, 'double_drugout': 7}
GENERATOR = 'gpt-6.1-sol medium plan/review; rareddie_control implementation by claude-opus-5-5'
NAMESPACE = 'rareddie_control'
ABLATION = ('accelerated_four no-path model with DSN InterGraphAttention, inter_projection and cross edges '
            'removed: original RareDDIE CSE order; same NAI/FIN, PVR, mu classifier, loss and protocol')
PROGRESS_EVERY = 100


def workspace_path(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Write/data workspace must remain inside {ROOT}: {path}')
    return path


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_checkpoint(path, checkpoint):
    # A killed writer must leave the preceding epoch checkpoint usable.
    temporary = path.with_name(path.name + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def top_k_accuracy(y, probabilities, k=5):
    # Report-only metric from the exported probabilities; never used for selection.
    top = np.argsort(-probabilities, axis=1, kind='stable')[:, :k]
    return float((top == y[:, None]).any(1).mean())


@torch.no_grad()
def evaluate(model, loader, device, config, beta):
    model.eval()
    totals = {name: 0. for name in ['ce', 'rec', 'kl', 'loss']}
    probabilities, targets, original_rows = [], [], []
    # Reconstruction uses normal Gaussian samples; evaluation sampling never
    # changes training RNG. Classifier always uses deterministic mu.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(config['seed'] + 10000)
        for raw in loader:
            batch = to_device(raw, device)
            output = model(batch)
            ce = F.cross_entropy(output['logits'], batch['label'])
            loss = ce + config['reconstruction_weight'] * output['rec'] + beta * output['kl']
            count = len(batch['label'])
            for name, value in [('ce', ce), ('rec', output['rec']), ('kl', output['kl']), ('loss', loss)]:
                totals[name] += float(value) * count
            probabilities.append(output['logits'].softmax(-1).cpu().numpy())
            targets.append(raw['label'].numpy())
            original_rows.append(raw['row_index'].numpy())
    n = len(loader.dataset)
    totals = {k: v / n for k, v in totals.items()}
    probs, y = np.concatenate(probabilities), np.concatenate(targets)
    totals['macro_f1_85'] = validation_selection_metric(y, probs)
    totals['kl_beta'] = beta
    return totals, probs, y, np.concatenate(original_rows)


def export_test(model, dataset, loader, device, config, output, best):
    totals, probabilities, y, original_rows = evaluate(model, loader, device, config, config['kl_weight'])
    metrics, per_class, confusion = classification_metrics(y, probabilities)
    metrics.update({'top5_accuracy': top_k_accuracy(y, probabilities), 'loss_terms': totals,
                    'checkpoint': 'best.pt', 'selected_epoch': best['epoch'],
                    'selected_validation_macro_f1_85': best['best_metric'], 'selection': 'validation only'})
    frame = dataset.frame.copy()
    assert np.array_equal(original_rows, dataset.rows)
    frame.insert(0, 'frozen_row_index', original_rows)
    frame['predicted_event_id'] = probabilities.argmax(1)
    frame['predicted_probability'] = probabilities.max(1)
    frame.to_parquet(output / 'test_predictions.parquet', index=False)
    np.savez_compressed(output / 'test_probabilities.npz', probabilities=probabilities,
                        event_ids=np.arange(85), labels=y, frozen_row_indices=original_rows)
    events = pd.read_csv(workspace_path(config['prepared']) / 'ddi_events.csv')
    per_class.merge(events, on='event_id', how='left').to_csv(output / 'test_per_class.csv', index=False)
    pd.DataFrame(confusion, index=np.arange(85), columns=np.arange(85)).rename_axis('true_event_id').to_csv(output / 'test_confusion.csv')
    save_json(output / 'test_metrics.json', metrics)
    print(json.dumps({'phase': 'best_checkpoint_test', **metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fresh', action='store_true')
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(workspace_path(args.config).read_text())
    # This is checked before the first CUDA operation. Only the assigned one
    # physical device is visible, including RNG and allocator operations.
    assigned = GPU[config['protocol']]
    if config['gpu'] != assigned or os.environ.get('CUDA_VISIBLE_DEVICES') != str(assigned):
        raise ValueError(f'This arm requires CUDA_VISIBLE_DEVICES={assigned} and config gpu={assigned}')
    torch.set_num_threads(config['cpu_threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = torch.device('cuda:0')
    prepared = workspace_path(config['prepared'])
    output_dir = workspace_path(config['output'])
    # Own output only; DSN/accelerated/production checkpoints are never read.
    if not output_dir.is_relative_to(OUTPUTS):
        raise ValueError(f'Output must remain inside {OUTPUTS}')
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh and any((output_dir / n).exists() for n in ['last.pt', 'best.pt', 'history.jsonl']):
        raise ValueError('--fresh requires an empty output; recovery uses --resume or --test-only')
    if not args.fresh:
        own = torch.load(output_dir / ('best.pt' if args.test_only else 'last.pt'), map_location='cpu', weights_only=False)
        if own['config'].get('namespace') != NAMESPACE or Path(own['config']['output']) != output_dir:
            raise ValueError('Checkpoint is not this rareddie_control run')
    saved = own if args.resume else None
    if args.test_only:
        assert own['config']['gpu'] == assigned and own['config']['protocol'] == config['protocol']
        # Evaluation restores the authoritative training configuration.
        config = own['config']
        del own
    if args.fresh:
        # Seeded state immediately before model construction (exact, restorable).
        torch.save(rng_state(), output_dir / 'initial_rng.pt')
    if not args.test_only:
        config.update({'prepared': str(prepared), 'output': str(output_dir), 'generator': GENERATOR,
                       'namespace': NAMESPACE, 'ablation': ABLATION, 'with_path': False,
                       'selection_metric': 'fixed85 validation macroF1', 'test_selection': False,
                       'source_provenance': json.loads((ROOT / 'docs/source_provenance.json').read_text())})
    preparation = json.loads((prepared / 'preparation_report.json').read_text())
    assert config['nai_cap'] == preparation['nai']['cap']
    audit = json.loads((prepared / 'audit_report.json').read_text())
    assert audit['protocols'][config['protocol']]['all_rows_exact']
    assets = torch.load(prepared / 'assets.pt', map_location='cpu', weights_only=False)
    model = RareDDIEControl(assets).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    collate = Collator(assets)
    datasets = {split: DDIDataset(prepared, config['protocol'], split) for split in ['train', 'val']}
    loaders = {split: DataLoader(dataset, batch_size=config['batch_size'], shuffle=split == 'train',
                                 num_workers=config['num_workers'], collate_fn=collate, pin_memory=True,
                                 persistent_workers=False)
               for split, dataset in datasets.items()}
    if args.fresh:
        config['planned_updates'] = config['epochs'] * len(loaders['train'])
        config['kl_warmup_updates'] = max(1, math.ceil(config['kl_warmup_fraction'] * config['planned_updates']))
        (output_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        save_json(output_dir / 'model_inventory.json', parameter_inventory(model))
    if saved is not None and saved['config'] != config:
        raise ValueError('Resume configuration differs from last.pt; preserve batch size and schedule')
    warmup_updates = config['kl_warmup_updates']
    best_metric, bad_epochs, update, start_epoch = -1., 0, 0, 1
    if args.resume:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        # Construct model/loaders BEFORE restoring RNG. With nonpersistent
        # workers, each epoch consumes the same iterator base seed on resume;
        # there are no random transforms. Shuffling uses this restored RNG.
        restore_rng(saved['rng'])
        best_metric, bad_epochs, update = saved['best_metric'], saved['bad_epochs'], saved['update']
        start_epoch = saved['epoch'] + 1
        del saved
    mode_name = 'fresh' if args.fresh else 'resume' if args.resume else 'test-only'
    will_train = not args.test_only and bad_epochs < config['patience'] and start_epoch <= config['epochs']
    # Liveness/start line only (device query and print; no RNG use).
    print(json.dumps({'phase': 'start', 'mode': mode_name,
                      'start_epoch': start_epoch, 'update': update, 'best_metric': best_metric,
                      'bad_epochs': bad_epochs, 'epochs': config['epochs'], 'patience': config['patience'],
                      'will_train': will_train, 'planned_updates': config['planned_updates'],
                      'kl_warmup_updates': warmup_updates, 'train_batches': len(loaders['train']),
                      'train_rows': len(datasets['train']), 'validation_rows': len(datasets['val']),
                      'batch_size': config['batch_size'], 'seed': config['seed'], 'output': str(output_dir),
                      'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'device_uuid': str(torch.cuda.get_device_properties(device).uuid), 'pid': os.getpid(),
                      'started_at': time.time()}), flush=True)
    start_update = update
    if will_train:
        for epoch in range(start_epoch, config['epochs'] + 1):
            started = time.time()
            model.train()
            totals = {name: 0. for name in ['ce', 'rec', 'kl', 'loss', 'kl_weighted']}
            count = 0
            for raw in loaders['train']:
                batch = to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                result = model(batch)
                ce = F.cross_entropy(result['logits'], batch['label'])
                beta = config['kl_weight'] * min(1., (update + 1) / warmup_updates)
                loss = ce + config['reconstruction_weight'] * result['rec'] + beta * result['kl']
                loss.backward()
                # Actual full-training first-update loss terms and gradients.
                if update == 0:
                    grads = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
                    save_json(output_dir / 'first_update.json', {
                        'labels': len(batch['label']),
                        'loss_terms': {'ce': float(ce), 'rec': float(result['rec']), 'kl': float(result['kl']),
                                       'beta': beta, 'loss': float(loss)},
                        'gradient_norms': {group: float(torch.stack([g.detach().square().sum() for n, g in grads.items()
                                                                     if g is not None and n.startswith(prefixes)]).sum().sqrt())
                                           for group, prefixes in GROUPS.items()},
                        'all_gradients_finite': all(bool(torch.isfinite(g).all()) for g in grads.values() if g is not None),
                        # Original forward never consumes the last block's norm2 output.
                        'trainable_without_gradient': [n for n, g in grads.items() if g is None]})
                optimizer.step()
                n = len(batch['label'])
                for name, value in [('ce', ce), ('rec', result['rec']), ('kl', result['kl']),
                                    ('loss', loss), ('kl_weighted', beta * result['kl'])]:
                    totals[name] += float(value.detach()) * n
                count += n
                update += 1
                if update == start_update + 1 or update % PROGRESS_EVERY == 0:
                    # From already-synchronized running totals; no RNG use.
                    print(json.dumps({'phase': 'train_progress', 'epoch': epoch, 'update': update,
                                      'epoch_rows': count, 'epoch_seconds': time.time() - started,
                                      'beta': beta, 'running_ce': totals['ce'] / count,
                                      'running_loss': totals['loss'] / count}), flush=True)
            train = {k: v / count for k, v in totals.items()}
            train['kl_beta_final'] = beta
            val, val_probabilities, val_y, _ = evaluate(model, loaders['val'], device, config, beta)
            val['accuracy'] = float((val_probabilities.argmax(1) == val_y).mean())
            del val_probabilities, val_y
            improved = val['macro_f1_85'] > best_metric
            if improved:
                best_metric, bad_epochs = val['macro_f1_85'], 0
            else:
                bad_epochs += 1
            record = {'epoch': epoch, 'update': update, 'train': train, 'validation': val,
                      'best_validation_macro_f1_85': best_metric, 'improved': improved,
                      'bad_epochs': bad_epochs, 'seconds': time.time() - started}
            with (output_dir / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record, allow_nan=False), flush=True)
            checkpoint = {'epoch': epoch, 'update': update, 'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(), 'config': config, 'best_metric': best_metric,
                          'bad_epochs': bad_epochs, 'metrics': record, 'rng': rng_state()}
            if improved:
                save_checkpoint(output_dir / 'best.pt', checkpoint)
            save_checkpoint(output_dir / 'last.pt', checkpoint)
            if bad_epochs >= config['patience']:
                break
    # Test is constructed and read by the training process only after validation
    # selection has finished.
    best = torch.load(output_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(best['model'])
    test_data = DDIDataset(prepared, config['protocol'], 'test')
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config['num_workers'], collate_fn=collate, pin_memory=True)
    export_test(model, test_data, test_loader, device, config, output_dir, best)
    (output_dir / 'test_config.yaml').write_text(yaml.safe_dump(
        {**best['config'], 'evaluation_batch_size': config['batch_size'], 'evaluation_checkpoint': 'best.pt'},
        sort_keys=False))
    save_json(output_dir / 'test_complete.json', {
        'checkpoint': 'best.pt', 'selected_epoch': best['epoch'], 'selected_update': best['update'],
        'rows': len(test_data), 'completed_at': time.time()})


if __name__ == '__main__':
    main()

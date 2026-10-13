"""GPU3 speed benchmark copy: original training, capped at epoch_cap full
train+validation epochs of the ORIGINAL epochs-based KL schedule. No test."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from .data import DDIDataset, Collator, to_device
from .metrics import validation_selection_metric
from .model import DSNRareDDI
from .status import update_status

ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / 'benchmarks/gpu3_speed/outputs'
GPU = 3
HISTORICAL_ORIGINAL_EPOCH_SECONDS = (1498, 1534)


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


@torch.no_grad()
def evaluate(model, loader, device, config, beta):
    model.eval()
    totals = {name: 0. for name in ['ce', 'rec', 'kl', 'loss']}
    probabilities, targets, original_rows, masks = [], [], [], []
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
            masks.append(output['path_mask'].cpu().numpy())
    n = len(loader.dataset)
    totals = {k: v / n for k, v in totals.items()}
    probs, y = np.concatenate(probabilities), np.concatenate(targets)
    totals['macro_f1_85'] = validation_selection_metric(y, probs)
    totals['accuracy'] = float((probs.argmax(1) == y).mean())
    totals['kl_beta'] = beta
    return totals, probs, y, np.concatenate(original_rows), np.concatenate(masks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(workspace_path(args.config).read_text())
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    # This is checked before the first CUDA operation. Only physical GPU3 is
    # visible, including RNG and allocator operations.
    if config['gpu'] != GPU or os.environ.get('CUDA_VISIBLE_DEVICES') != str(GPU):
        raise ValueError(f'This benchmark requires CUDA_VISIBLE_DEVICES={GPU} and config gpu={GPU}')
    # The cap only stops the loop; the KL schedule still uses config epochs.
    if not 1 <= config['epoch_cap'] <= config['epochs']:
        raise ValueError('epoch_cap must be within the original epochs')
    torch.set_num_threads(config['cpu_threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = torch.device('cuda:0')
    prepared = workspace_path(config['prepared'])
    output_dir = workspace_path(config['output'])
    # Independent benchmark output only; production outputs are never written.
    if not output_dir.is_relative_to(OUTPUTS):
        raise ValueError(f'Benchmark output must remain inside {OUTPUTS}')
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = torch.load(output_dir / 'last.pt', map_location='cpu', weights_only=False) if args.resume else None
    if not args.resume and (output_dir / 'last.pt').exists():
        raise ValueError('Existing training output: use --resume or a new output directory')
    loader_batch = config['batch_size']
    # Save the fully resolved settings (including memory-driven batch override).
    config.update({'prepared': str(prepared), 'output': str(output_dir),
                   'generator': 'gpt-6.1-sol high; GPU3 packed-attention copy by claude-opus-5-5',
                   'selection_metric': 'fixed85 validation macroF1', 'test_selection': False,
                   'source_provenance': json.loads((ROOT / 'docs/source_provenance.json').read_text())})
    preparation = json.loads((prepared / 'preparation_report.json').read_text())
    assert config['nai_cap'] == preparation['nai']['cap']
    assert config['path_depth'] == preparation['paths']['depth']
    assert config['path_meeting_cap'] == preparation['paths']['meeting_cap']
    assert config['path_arm_cap'] == preparation['paths']['arm_cap']
    # Full-data audit is a preparation requirement, not a toy model probe.
    audit = json.loads((prepared / 'audit_report.json').read_text())
    assert audit['all_cached_real_edges_found_in_source'] and audit['all_pairs_depth4_reachability_equals_explicit_mask']
    assets = torch.load(prepared / 'assets.pt', map_location='cpu', weights_only=False)
    model = DSNRareDDI(assets, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    collate = Collator(assets, prepared, config['with_path'])
    datasets = {split: DDIDataset(prepared, config['protocol'], split) for split in ['train', 'val']}
    loaders = {split: DataLoader(dataset, batch_size=loader_batch, shuffle=split == 'train',
                                 num_workers=config['num_workers'], collate_fn=collate, pin_memory=True,
                                 persistent_workers=False)
               for split, dataset in datasets.items()}
    planned_updates = config['epochs'] * len(loaders['train'])
    warmup_updates = max(1, math.ceil(config['kl_warmup_fraction'] * planned_updates))
    if saved is not None:
        planned_updates = saved['config']['planned_updates']
        warmup_updates = saved['config']['kl_warmup_updates']
    config['planned_updates'], config['kl_warmup_updates'] = planned_updates, warmup_updates
    if saved is not None and saved['config'] != config:
        raise ValueError('Resume configuration differs from last.pt; preserve batch size and schedule')
    (output_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({'phase': 'start', 'train_rows': len(datasets['train']), 'validation_rows': len(datasets['val']),
                      'train_batches': len(loaders['train']), 'epochs_schedule': config['epochs'],
                      'epoch_cap': config['epoch_cap'], 'planned_updates': planned_updates,
                      'kl_warmup_updates': warmup_updates, 'resume': args.resume}), flush=True)
    save_json(output_dir / 'model_inventory.json', {
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'fin_trainable': model.fin_entities.weight.requires_grad and model.fin_relations.weight.requires_grad,
        'fin_entity_table': list(model.fin_entities.weight.shape), 'fin_relation_table': list(model.fin_relations.weight.shape),
        'classes': 85, 'latent': 128, 'pair_input': 514 if config['with_path'] else 256,
        'pvr_continuous_decoder': 512 if config['with_path'] else 256,
        'pvr_branch_normalization': 'affine-free layer_norm per 128; absent directions zero',
        'pvr_reconstruction': 'valid-branch mean per sample, then sample mean; detached continuous targets',
        'path_depth': config['path_depth'] if config['with_path'] else 0,
        'topology_cache': str(prepared), 'learned_state_cache': False})
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
    # Exactly epoch_cap full epochs; no early stopping inside the cap.
    for epoch in range(start_epoch, config['epoch_cap'] + 1):
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
            # Capture actual full-training first-step gradient norms for
            # coordinator auditing, without a separate probe experiment.
            if update == 0:
                groups = {'cse': model.cse, 'nai': model.nai, 'pvr': model.pvr,
                          'classifier': model.classifier, 'fin_entities': model.fin_entities,
                          'fin_relations': model.fin_relations}
                if config['with_path']:
                    groups['path'] = model.path
                save_json(output_dir / 'first_update_gradients.json', {
                    name: float(torch.stack([p.grad.detach().square().sum() for p in module.parameters()
                                             if p.grad is not None]).sum().sqrt()) for name, module in groups.items()})
            optimizer.step()
            n = len(batch['label'])
            for name, value in [('ce', ce), ('rec', result['rec']), ('kl', result['kl']),
                                ('loss', loss), ('kl_weighted', beta * result['kl'])]:
                totals[name] += float(value.detach()) * n
            count += n
            update += 1
            if update % 100 == 0:
                # Liveness from already-synchronized running totals; no RNG use.
                progress = {'phase': 'train_progress', 'epoch': epoch, 'update': update, 'epoch_rows': count,
                            'epoch_seconds': time.time() - started, 'running_ce': totals['ce'] / count,
                            'running_loss': totals['loss'] / count}
                print(json.dumps(progress), flush=True)
                update_status('training', progress=progress)
        train = {k: v / count for k, v in totals.items()}
        train['kl_beta_final'] = beta
        train_seconds = time.time() - started
        progress = {'phase': 'validation', 'epoch': epoch, 'update': update, 'train_seconds': train_seconds}
        print(json.dumps(progress), flush=True)
        update_status('training', progress=progress)
        val, _, _, _, _ = evaluate(model, loaders['val'], device, config, beta)
        validation_seconds = time.time() - started - train_seconds
        improved = val['macro_f1_85'] > best_metric
        if improved:
            best_metric, bad_epochs = val['macro_f1_85'], 0
        else:
            bad_epochs += 1
        record = {'epoch': epoch, 'update': update, 'train': train, 'validation': val,
                  'best_validation_macro_f1_85': best_metric, 'improved': improved,
                  'bad_epochs': bad_epochs, 'train_seconds': train_seconds,
                  'validation_seconds': validation_seconds, 'seconds': train_seconds + validation_seconds,
                  'train_rows': count, 'validation_rows': len(datasets['val'])}
        record['historical_original_epoch_seconds'] = list(HISTORICAL_ORIGINAL_EPOCH_SECONDS)
        record['speed_ratio_vs_historical'] = [s / record['seconds'] for s in HISTORICAL_ORIGINAL_EPOCH_SECONDS]
        with (output_dir / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record, allow_nan=False), flush=True)
        checkpoint = {'epoch': epoch, 'update': update, 'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'config': config, 'best_metric': best_metric,
                      'bad_epochs': bad_epochs, 'metrics': record, 'rng': rng_state()}
        if improved:
            save_checkpoint(output_dir / 'best.pt', checkpoint)
        save_checkpoint(output_dir / 'last.pt', checkpoint)
        update_status('training', progress={'epochs_completed': epoch, 'epoch_cap': config['epoch_cap'],
                                            'last_epoch_seconds': record['seconds']})


if __name__ == '__main__':
    main()

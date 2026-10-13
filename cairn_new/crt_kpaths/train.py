"""Step2: end-to-end DSN-DDI + EmerGNN + RareDDIE (with_path) on K-Paths topology.

main() is accelerated_four/dsn/train.py main() with ONLY these differences
(crt_kpaths/review/step2_train_vs_original.diff): GPU5 single / GPU6 double,
outputs under crt_kpaths/outputs, the K-Paths cache (same pair indices and array
layout) replaces the old route cache in the Collator and its preparation
asserts, the seed-42 initial state is saved, test_metrics.json gains
top5_accuracy and the train-and-test-positive-class diagnostic (crt.evaluation
on the exported probabilities) plus an 85-row test_class_counts.csv, and metadata. Model, loss, data,
optimizer, KL schedule, RNG consumption, epoch plan, selection and test export
are imported unchanged; no original best/last checkpoint is loaded.
"""
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
from accelerated_four.dsn.data import DDIDataset, Collator, to_device
from accelerated_four.dsn.model import DSNRareDDI
from accelerated_four.dsn.train import (PROGRESS_EVERY, OPTIMIZATION, evaluate, export_test, restore_rng,
                                        rng_state, save_checkpoint, save_json, workspace_path)
from .crt import step2_test_additions

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'crt_kpaths/outputs'
GPU = {'single_drugout': 5, 'double_drugout': 6}
GENERATOR = 'claude-opus-5-5 crt_kpaths Step2 (accelerated_four trainer, K-Paths topology)'
NAMESPACE = 'crt_kpaths_step2'
TOPOLOGY = ('K-Paths K=10 L=4 (official default L=3; adapted to the 4-layer path branch), complete non-DDI '
            'DRKG + inverses, full-signature hops expanded to atomic relations, candidate budget 1000')
INITIALIZATION = ('replayed original trainer seed 42: random/numpy/torch.manual_seed(42) then '
                  'DSNRareDDI(assets, config) on CPU; the original runs kept no initial state')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    if args.resume and args.test_only:
        parser.error('Use either --resume or --test-only')
    config = yaml.safe_load(workspace_path(args.config).read_text())
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    # This is checked before the first CUDA operation. Only the assigned one
    # physical device is visible, including RNG and allocator operations.
    assigned = GPU[config['protocol']]
    assert config['with_path']
    if config['gpu'] != assigned or os.environ.get('CUDA_VISIBLE_DEVICES') != str(assigned):
        raise ValueError(f'This arm requires CUDA_VISIBLE_DEVICES={assigned} and config gpu={assigned}')
    torch.set_num_threads(config['cpu_threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = torch.device('cuda:0')
    prepared = workspace_path(config['prepared'])
    output_dir = workspace_path(config['output'])
    path_cache = workspace_path(config['path_cache'])
    # Independent replica output only; production outputs are never written.
    if not output_dir.is_relative_to(OUTPUTS):
        raise ValueError(f'Output must remain inside {OUTPUTS}')
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = torch.load(output_dir / 'last.pt', map_location='cpu', weights_only=False) if args.resume else None
    if not args.resume and not args.test_only and (output_dir / 'last.pt').exists():
        raise ValueError('Existing training output: use --resume or a new output directory')
    if args.test_only:
        selected = torch.load(output_dir / 'best.pt', map_location='cpu', weights_only=False)
        assert selected['config']['gpu'] == assigned and selected['config']['protocol'] == config['protocol']
        # Evaluation restores the authoritative training configuration and keeps
        # config.yaml intact, even when the evaluation batch is overridden.
        config = selected['config']
        del selected
    loader_batch = args.batch_size if args.test_only and args.batch_size else config['batch_size']
    # Save the fully resolved settings (including memory-driven batch override).
    if not args.test_only:
        config.update({'prepared': str(prepared), 'output': str(output_dir), 'generator': GENERATOR,
                       'namespace': NAMESPACE, 'optimization': OPTIMIZATION, 'topology': TOPOLOGY,
                       'path_cache': str(path_cache), 'initialization': INITIALIZATION,
                       'selection_metric': 'fixed85 validation macroF1', 'test_selection': False,
                       'source_provenance': json.loads((ROOT / 'docs/source_provenance.json').read_text())})
    preparation = json.loads((prepared / 'preparation_report.json').read_text())
    assert config['nai_cap'] == preparation['nai']['cap']
    assert config['path_depth'] == preparation['paths']['depth']
    # K-Paths topology replaces the old route cache; same pair indexing.
    kpaths = json.loads((path_cache / 'report.json').read_text())
    assert kpaths['complete'] and kpaths['pairs'] == len(np.load(prepared / 'pairs.npy'))
    assert (config['kpaths_k'], config['kpaths_max_length'], config['kpaths_candidate_budget']) == (
        kpaths['k'], kpaths['max_length'], kpaths['candidate_budget'])
    assert config['kpaths_max_length'] <= config['path_depth']
    cache_check = json.loads((ROOT / 'crt_kpaths/evidence/kpaths_cache_check.json').read_text())
    assert cache_check['path_mask_equals_connected_depth4_reachability'] and cache_check['pairs'] == kpaths['pairs']
    assets = torch.load(prepared / 'assets.pt', map_location='cpu', weights_only=False)
    model = DSNRareDDI(assets, config).to(device)
    if not args.resume and not args.test_only:
        # Shared initial state: the original trainer's seed-42 construction
        # replayed (no extra RNG use); this run's own record, nothing loaded.
        save_checkpoint(output_dir / 'initial_state.pt', {
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'source': INITIALIZATION})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    collate = Collator(assets, path_cache, config['with_path'])
    datasets = {split: DDIDataset(prepared, config['protocol'], split) for split in ['train', 'val']}
    loaders = {split: DataLoader(dataset, batch_size=loader_batch, shuffle=split == 'train',
                                 num_workers=config['num_workers'], collate_fn=collate, pin_memory=True,
                                 persistent_workers=False)
               for split, dataset in datasets.items()}
    planned_updates = config['planned_updates'] if args.test_only else config['epochs'] * len(loaders['train'])
    warmup_updates = config['kl_warmup_updates'] if args.test_only else max(1, math.ceil(config['kl_warmup_fraction'] * planned_updates))
    if saved is not None:
        planned_updates = saved['config']['planned_updates']
        warmup_updates = saved['config']['kl_warmup_updates']
    config['planned_updates'], config['kl_warmup_updates'] = planned_updates, warmup_updates
    if saved is not None and saved['config'] != config:
        raise ValueError('Resume configuration differs from last.pt; preserve batch size and schedule')
    if not args.test_only:
        (output_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
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
        'topology_cache': str(path_cache), 'learned_state_cache': False})
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
    will_train = not args.test_only and bad_epochs < config['patience'] and start_epoch <= config['epochs']
    # Liveness/start line only (device query and print; no RNG use).
    print(json.dumps({'phase': 'start', 'mode': 'test-only' if args.test_only else 'resume' if args.resume else 'fresh',
                      'start_epoch': start_epoch, 'update': update, 'best_metric': best_metric,
                      'bad_epochs': bad_epochs, 'epochs': config['epochs'], 'patience': config['patience'],
                      'will_train': will_train, 'planned_updates': planned_updates,
                      'kl_warmup_updates': warmup_updates, 'train_batches': len(loaders['train']),
                      'train_rows': len(datasets['train']), 'validation_rows': len(datasets['val']),
                      'batch_size': loader_batch, 'output': str(output_dir),
                      'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'device_uuid': str(torch.cuda.get_device_properties(device).uuid), 'pid': os.getpid(),
                      'started_at': time.time()}), flush=True)
    start_update = update
    if not args.test_only and bad_epochs < config['patience']:
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
                if update == start_update + 1 or update % PROGRESS_EVERY == 0:
                    # From already-synchronized running totals; no RNG use.
                    print(json.dumps({'phase': 'train_progress', 'epoch': epoch, 'update': update,
                                      'epoch_rows': count, 'epoch_seconds': time.time() - started,
                                      'beta': beta, 'running_ce': totals['ce'] / count,
                                      'running_loss': totals['loss'] / count}), flush=True)
            train = {k: v / count for k, v in totals.items()}
            train['kl_beta_final'] = beta
            val, val_probabilities, val_y, _, _ = evaluate(model, loaders['val'], device, config, beta)
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
    # selection has finished. Preprocessing used its endpoints, never labels.
    best = torch.load(output_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(best['model'])
    test_data = DDIDataset(prepared, config['protocol'], 'test')
    test_loader = DataLoader(test_data, batch_size=loader_batch, shuffle=False,
                             num_workers=config['num_workers'], collate_fn=collate, pin_memory=True)
    export_test(model, test_data, test_loader, device, config, output_dir, best)
    metrics, counts = step2_test_additions(output_dir, {
        'train': datasets['train'].y, 'val': datasets['val'].y, 'test': test_data.y})
    save_json(output_dir / 'test_metrics.json', metrics)
    counts.to_csv(output_dir / 'test_class_counts.csv', index=False)
    (output_dir / 'test_config.yaml').write_text(yaml.safe_dump(
        {**best['config'], 'evaluation_batch_size': loader_batch, 'evaluation_checkpoint': 'best.pt'}, sort_keys=False))
    save_json(output_dir / 'test_complete.json', {
        'checkpoint': 'best.pt', 'selected_epoch': best['epoch'], 'selected_update': best['update'],
        'rows': len(test_data), 'completed_at': time.time()})


if __name__ == '__main__':
    main()

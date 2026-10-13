"""Full DDI training, fixed85 validation selection, selected-checkpoint test export.

pairout_cellout copy of the accelerated_four trainer: the same reviewed GPU3
packed model.py/data.py, ALWAYS with the EmerGNN path branch, protocols pairout
(GPU6) and cellout (GPU7) on the identical frozen PairOut DDI split, outputs
only under pairout_cellout/outputs. Strict gate on the new pairout_cellout
preparation/audit reports. --fresh is the initial launch only; --resume and
--test-only continue this run's own checkpoints with the exact config.
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
from .data import DDIDataset, Collator, to_device
from .metrics import classification_metrics, validation_selection_metric
from .model import DSNRareDDI

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / 'pairout_cellout/outputs'
PREPARED = ROOT / 'pairout_cellout/prepared'
GPU = {'pairout': 6, 'cellout': 7}
GENERATOR = 'gpt-6.1-sol medium plan/review; pairout_cellout implementation by claude-opus-5-5'
NAMESPACE = 'pairout_cellout'
OPTIMIZATION = ('GPU3-reviewed packed cross-pair inter-graph attention: one GATConv call per direction, '
                'per-pair local self loops built by the collator, add_self_loops=False')
PROGRESS_EVERY = 100
# Authorized schedule for THESE jobs (latest human update: 150 maximum epochs,
# overriding 100): 151508 rows / 128 -> 1184 batches, 150 * 1184 = 177600
# planned updates, KL warmup ceil(0.2 * 177600) = 35520 (30 epochs), patience 15.
SCHEDULE = {'epochs': 150, 'patience': 15, 'kl_warmup_fraction': .2,
            'planned_updates': 177600, 'kl_warmup_updates': 35520}
TRAIN_BATCHES = 1184
# Parameter groups for the actual first-update diagnostics (all must receive
# finite nonzero gradients); the last CSE block's norm2 output is never consumed.
GROUPS = {'fin_entities': ('fin_entities.',), 'fin_relations': ('fin_relations.',), 'cse': ('cse.',),
          'nai': ('nai.',), 'path': ('path.',), 'pvr': ('pvr.',), 'classifier': ('classifier.',)}
AUDIT_CHECKS = ['all_cached_real_edges_found_in_source', 'source_fact_multiplicity_preserved',
                'all_real_edges_have_inverse', 'endpoint_mapping_exact',
                'all_pairs_depth4_reachability_equals_explicit_mask', 'missing_paths_have_no_edges',
                'explicit_self_count_exact', 'no_compound_compound_background_facts', 'pairout_cellout_frames_identical',
                'fin_tables_exact', 'nai_table_equal_to_data_prepared', 'original_data_prepared_unchanged', 'passed']


def preparation_gate(prepared, config):
    """Strict new-cohort preparation + full audit requirement (no toy probe)."""
    if prepared != PREPARED:
        raise ValueError(f'prepared must be {PREPARED}')
    preparation = json.loads((prepared / 'preparation_report.json').read_text())
    audit = json.loads((prepared / 'audit_report.json').read_text())
    assert preparation['namespace'] == audit['namespace'] == NAMESPACE
    failed = [name for name in AUDIT_CHECKS if audit.get(name) is not True]
    if failed:
        raise ValueError(f'Full preparation audit not passed: {failed}')
    newest = max((p for p in prepared.rglob('*') if p.is_file() and p.name != 'audit_report.json'),
                 key=lambda p: p.stat().st_mtime)
    if (prepared / 'audit_report.json').stat().st_mtime < newest.stat().st_mtime:
        raise ValueError(f'audit_report.json is older than {newest}; rerun pairout_cellout.audit')
    protocol = audit['protocols'][config['protocol']]
    assert all(protocol[k] is True for k in ['all_rows_exact', 'all_columns_order_dtypes_exact', 'event_names_exact',
                                             'global_drug_ids_exact', 'pair_index_exact', 'split_pair_leakage_zero'])
    assert config['nai_cap'] == preparation['nai']['cap']
    assert config['path_depth'] == preparation['paths']['depth']
    assert config['path_meeting_cap'] == preparation['paths']['meeting_cap']
    assert config['path_arm_cap'] == preparation['paths']['arm_cap']
    return preparation, audit


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
    totals['kl_beta'] = beta
    return totals, probs, y, np.concatenate(original_rows), np.concatenate(masks)


def export_test(model, dataset, loader, device, config, output, best):
    totals, probabilities, y, original_rows, masks = evaluate(model, loader, device, config, config['kl_weight'])
    metrics, per_class, confusion = classification_metrics(y, probabilities)
    metrics.update({'top5_accuracy': top_k_accuracy(y, probabilities), 'loss_terms': totals,
                    'checkpoint': 'best.pt', 'selected_epoch': best['epoch'],
                    'selected_validation_macro_f1_85': best['best_metric'], 'selection': 'validation only'})
    frame = dataset.frame.copy()
    assert np.array_equal(original_rows, dataset.rows)
    frame.insert(0, 'frozen_row_index', original_rows)
    frame['predicted_event_id'] = probabilities.argmax(1)
    frame['predicted_probability'] = probabilities.max(1)
    if config['with_path']:
        frame['path_v_to_u_present'] = masks[:, 0]
        frame['path_u_to_v_present'] = masks[:, 1]
    frame.to_parquet(output / 'test_predictions.parquet', index=False)
    np.savez_compressed(output / 'test_probabilities.npz', probabilities=probabilities,
                        event_ids=np.arange(85), labels=y, frozen_row_indices=original_rows, path_mask=masks)
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
    # Full-path model only, on the assigned GPU. This is checked before the
    # first CUDA operation. Only the assigned one physical device is visible,
    # including RNG and allocator operations.
    if config['with_path'] is not True:
        raise ValueError('pairout_cellout always trains the full path model (with_path: true)')
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
    # Own output only; DSN/accelerated/GPU3/production checkpoints are never read.
    if not output_dir.is_relative_to(OUTPUTS):
        raise ValueError(f'Output must remain inside {OUTPUTS}')
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh and any((output_dir / n).exists() for n in ['last.pt', 'best.pt', 'history.jsonl']):
        raise ValueError('--fresh requires an empty output; recovery uses --resume or --test-only')
    if not args.fresh:
        own = torch.load(output_dir / ('best.pt' if args.test_only else 'last.pt'), map_location='cpu', weights_only=False)
        if own['config'].get('namespace') != NAMESPACE or Path(own['config']['output']) != output_dir:
            raise ValueError('Checkpoint is not this pairout_cellout run')
    saved = own if args.resume else None
    if args.test_only:
        assert own['config']['gpu'] == assigned and own['config']['protocol'] == config['protocol']
        # Evaluation restores the authoritative training configuration.
        config = own['config']
        del own
    if args.fresh:
        # Seeded state immediately before model construction (exact, restorable).
        torch.save(rng_state(), output_dir / 'initial_rng.pt')
    preparation, audit = preparation_gate(prepared, config)
    if not args.test_only:
        config.update({'prepared': str(prepared), 'output': str(output_dir), 'generator': GENERATOR,
                       'namespace': NAMESPACE, 'optimization': OPTIMIZATION,
                       'selection_metric': 'fixed85 validation macroF1', 'test_selection': False,
                       'cohort_source_identity': preparation['source_identity'],
                       'source_provenance': json.loads((ROOT / 'docs/source_provenance.json').read_text())})
    assets = torch.load(prepared / 'assets.pt', map_location='cpu', weights_only=False)
    model = DSNRareDDI(assets, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    collate = Collator(assets, prepared, config['with_path'])
    datasets = {split: DDIDataset(prepared, config['protocol'], split) for split in ['train', 'val']}
    loaders = {split: DataLoader(dataset, batch_size=config['batch_size'], shuffle=split == 'train',
                                 num_workers=config['num_workers'], collate_fn=collate, pin_memory=True,
                                 persistent_workers=False)
               for split, dataset in datasets.items()}
    # Audited schedule: 151508 rows / 128 -> 1184 batches, 177600 updates, KL warmup 35520.
    assert len(loaders['train']) == audit['schedule']['train_batches'] == TRAIN_BATCHES
    if args.fresh:
        config['planned_updates'] = config['epochs'] * len(loaders['train'])
        config['kl_warmup_updates'] = max(1, math.ceil(config['kl_warmup_fraction'] * config['planned_updates']))
        assert config['planned_updates'] == audit['schedule']['planned_updates']
        assert config['kl_warmup_updates'] == audit['schedule']['kl_warmup_updates']
        (output_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    if saved is not None and saved['config'] != config:
        raise ValueError('Resume configuration differs from last.pt; preserve batch size and schedule')
    # Every mode (fresh, resume, test-only) runs only the authorized schedule.
    if any(config[k] != v for k, v in SCHEDULE.items()):
        raise ValueError(f'Schedule must be {SCHEDULE}')
    planned_updates, warmup_updates = config['planned_updates'], config['kl_warmup_updates']
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
        'topology_cache': str(prepared), 'learned_state_cache': False,
        'groups': {g: sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefixes))
                   for g, prefixes in GROUPS.items()},
        'state_dict_keys': list(model.state_dict()),
        'frozen': [n for n, p in model.named_parameters() if not p.requires_grad]})
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
                      'batch_size': config['batch_size'], 'seed': config['seed'], 'with_path': config['with_path'],
                      'protocol': config['protocol'], 'output': str(output_dir),
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
                # Actual full-training first-update loss terms and gradients of
                # every group including path, without a separate probe experiment.
                if update == 0:
                    grads = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
                    graph = batch['path_graph']
                    save_json(output_dir / 'first_update.json', {
                        'labels': len(batch['label']),
                        'loss_terms': {'ce': float(ce), 'rec': float(result['rec']), 'kl': float(result['kl']),
                                       'beta': beta, 'loss': float(loss)},
                        'logits_finite': bool(torch.isfinite(result['logits']).all()),
                        'gradient_norms': {group: float(torch.stack([g.detach().square().sum() for n, g in grads.items()
                                                                     if g is not None and n.startswith(prefixes)]).sum().sqrt())
                                           for group, prefixes in GROUPS.items()},
                        'all_gradients_finite': all(bool(torch.isfinite(g).all()) for g in grads.values() if g is not None),
                        'trainable_without_gradient': [n for n, g in grads.items() if g is None],
                        'path_batch': {'nodes': int(graph['n_nodes']), 'edges': int(graph['edge_index'].shape[1]),
                                       'pairs_with_cached_path': int(graph['path_mask'].sum()),
                                       'present_directions_v_to_u_u_to_v': result['path_mask'].sum(0).tolist()}})
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
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config['num_workers'], collate_fn=collate, pin_memory=True)
    export_test(model, test_data, test_loader, device, config, output_dir, best)
    (output_dir / 'test_config.yaml').write_text(yaml.safe_dump(
        {**best['config'], 'evaluation_batch_size': config['batch_size'], 'evaluation_checkpoint': 'best.pt'}, sort_keys=False))
    save_json(output_dir / 'test_complete.json', {
        'checkpoint': 'best.pt', 'selected_epoch': best['epoch'], 'selected_update': best['update'],
        'rows': len(test_data), 'completed_at': time.time()})


if __name__ == '__main__':
    main()

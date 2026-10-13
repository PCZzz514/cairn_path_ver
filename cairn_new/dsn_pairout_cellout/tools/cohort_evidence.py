"""Small exact frame/source/cohort checks, model/data copy record and CPU inventory (read-only inputs).

No preparation, audit rerun, model forward or benchmark: the full pairout_cellout
preparation audit, the GPU3 numerical evidence for the unchanged packed DSN
model/data and the accelerated_four approval are reused. The model is only
constructed on CPU (no forward) to record its parameter inventory: DSN
GAT/projection present, path module NONE, the collator loads no path arrays,
every non-DSN group equal to the live pure RareDDIE control inventory and the
total equal to the accelerated_four no-path inventory.
Writes dsn_pairout_cellout/evidence/cohort_copy_evidence.json.
Usage: source dsn_pairout_cellout/env.sh; CUDA_VISIBLE_DEVICES='' $PYTHON_BIN -m dsn_pairout_cellout.tools.cohort_evidence
"""
import filecmp
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / 'dsn_pairout_cellout'
PREPARED = ROOT / 'pairout_cellout/prepared'
SOURCES = {'pairout': ROOT.parent / 'cairn_dev/data/processed/ddi.parquet',
           'cellout': ROOT.parent / 'cairn_dev/data/experiments/reference85/cellout/ddi.parquet'}
COUNTS = {'train': 151508, 'val': 18939, 'test': 18938}
CLASSES = {'train': 85, 'val': 82, 'test': 83}
ORIGINAL = ROOT / 'accelerated_four/dsn'
PRIOR_GPU3 = ROOT / 'benchmarks/gpu3_speed/evidence/gpu3_evidence.json'
GPU3_COPY = ROOT / 'benchmarks/gpu3_speed/dsn_gpu3'
PRIOR_APPROVALS = [ROOT / 'logs/gpu3_speed/review_approval.json', ROOT / 'logs/accelerated_four/review_approval.json']
# Read-only reference inventories: live pure control (same FIN assets) and the
# accelerated_four no-path model (same files, same FIN table shapes).
PURE_INVENTORY = ROOT / 'rareddie_pairout_cellout/outputs/pairout/model_inventory.json'
ACCELERATED_NO_PATH_INVENTORY = ROOT / 'accelerated_four/outputs/no_path_single_drugout/model_inventory.json'


def same_frame(a, b):
    return {'columns_and_order': a.columns.tolist() == b.columns.tolist(),
            'dtypes': a.dtypes.astype(str).tolist() == b.dtypes.astype(str).tolist(),
            'index': a.index.equals(b.index), 'values': a.equals(b)}


def main():
    sources = {arm: pd.read_parquet(path) for arm, path in SOURCES.items()}
    frames = {arm: pd.read_parquet(PREPARED / arm / 'ddi.parquet') for arm in SOURCES}
    pairs = {arm: np.load(PREPARED / arm / 'pair_indices.npy') for arm in SOURCES}
    assets = torch.load(PREPARED / 'assets.pt', map_location='cpu', weights_only=False)
    events = pd.read_csv(PREPARED / 'ddi_events.csv')
    report = {'sources': {arm: {'path': str(path), 'resolved': str(path.resolve())} for arm, path in SOURCES.items()},
              'columns': frames['pairout'].columns.tolist(),
              'dtypes': frames['pairout'].dtypes.astype(str).tolist(),
              'source_pairout_vs_cellout': same_frame(sources['pairout'], sources['cellout']),
              'prepared_pairout_vs_cellout': same_frame(frames['pairout'], frames['cellout']),
              'pair_indices_pairout_equal_cellout': bool(np.array_equal(pairs['pairout'], pairs['cellout'])),
              'arms': {}}
    graphs = assets['graphs']
    for arm, frame in frames.items():
        counts = frame.split.value_counts().to_dict()
        classes = {split: int(frame.event_id[frame.split.eq(split)].nunique()) for split in COUNTS}
        drugs = np.union1d(frame.drug_a.to_numpy(), frame.drug_b.to_numpy())
        report['arms'][arm] = {
            'prepared_vs_source': same_frame(frame, sources[arm]),
            'counts': counts, 'counts_exact': counts == COUNTS,
            'classes_present': classes, 'classes_exact': classes == CLASSES,
            'pair_indices_rows': len(pairs[arm]), 'pair_indices_rows_exact': len(pairs[arm]) == len(frame),
            'drugs': len(drugs),
            'all_drugs_have_graph': all(int(d) in graphs for d in drugs),
            'all_drugs_have_nai_row': bool(drugs.min() >= 0 and drugs.max() < len(assets['neighbor_entities'])),
            'event_ids_in_0_84': bool(frame.event_id.between(0, 84).all())}
    report['event_names_exact'] = (events.event_id.tolist() == list(range(85))
                                   and events.template.tolist() == assets['ddi_event_names']
                                   and assets['num_classes'] == 85)
    batches = math.ceil(COUNTS['train'] / 128)
    report['schedule'] = {'train_batches': batches, 'planned_updates': 150 * batches,
                          'kl_warmup_updates': math.ceil(.2 * 150 * batches),
                          'kl_warmup_epochs': math.ceil(.2 * 150 * batches) / batches}
    report['copy'] = {
        'model_py_byte_identical_to_accelerated_four': filecmp.cmp(CONTROL / 'dsn/model.py', ORIGINAL / 'model.py', shallow=False),
        'data_py_byte_identical_to_accelerated_four': filecmp.cmp(CONTROL / 'dsn/data.py', ORIGINAL / 'data.py', shallow=False),
        'accelerated_four_byte_identical_to_gpu3_verified': all(
            filecmp.cmp(ORIGINAL / name, GPU3_COPY / name, shallow=False) for name in ['model.py', 'data.py']),
        'prior_gpu3_evidence': str(PRIOR_GPU3.relative_to(ROOT)),
        'prior_gpu3_passed': json.loads(PRIOR_GPU3.read_text())['all_numerical_checks_passed'] is True,
        'prior_approvals': [str(p.relative_to(ROOT)) for p in PRIOR_APPROVALS],
        'prior_approved': all(json.loads(p.read_text())['approved'] is True for p in PRIOR_APPROVALS),
        'note': 'dsn/model.py+data.py are the GPU3-verified accelerated_four files; no new benchmark was run'}
    # CPU construction only (no forward), exactly as the trainer builds them.
    from dsn_pairout_cellout.dsn.data import Collator
    from dsn_pairout_cellout.dsn.model import DSNRareDDI
    from dsn_pairout_cellout.inventory import parameter_inventory
    collator = Collator(assets, PREPARED, False)
    inventory = parameter_inventory(DSNRareDDI(assets, {'with_path': False}), collator)
    pure, accelerated = json.loads(PURE_INVENTORY.read_text()), json.loads(ACCELERATED_NO_PATH_INVENTORY.read_text())
    report['inventory'] = {k: v for k, v in inventory.items() if k != 'tensors'}
    report['inventory_checks'] = {
        'non_dsn_groups_equal_pure_control': all(inventory['groups'][g] == n for g, n in pure['groups'].items()),
        'dsn_groups_nonzero': inventory['groups']['dsn_inter_attention'] > 0 and inventory['groups']['inter_projection'] > 0,
        'total_equals_pure_plus_dsn': inventory['parameters'] == pure['parameters'] + inventory['groups']['dsn_inter_attention']
        + inventory['groups']['inter_projection'],
        'total_equals_accelerated_four_no_path': inventory['parameters'] == accelerated['parameters']
        and inventory['trainable_parameters'] == accelerated['trainable_parameters'],
        'frozen_only_readout_select': inventory['frozen'] == pure['frozen'],
        'two_dsn_gatconvs_without_auto_self_loops': len(inventory['dsn_inter_attention_gatconvs']) == 2 and all(
            not g['add_self_loops'] for g in inventory['dsn_inter_attention_gatconvs'].values()),
        'inter_projection_present': inventory['inter_projection_present'],
        'path_module_none': inventory['path_module'] == 'NONE' and inventory['path_parameters'] == 0
        and inventory['with_path'] is False,
        'collator_loads_no_path_arrays': not inventory['collator_path_arrays_loaded'],
        'pvr_no_path_input': inventory['pair_input'] == 256 and inventory['pvr_presence_dim'] == 0
        and inventory['latent'] == 128,
        'classifier_same_as_pure': inventory['classifier'] == pure['classifier'],
        'fin_trainable_same_tables': inventory['fin_trainable'] is True
        and inventory['fin_entity_table'] == accelerated['fin_entity_table']
        and inventory['fin_relation_table'] == accelerated['fin_relation_table']}
    report['ddi_split_note'] = ('same frozen PairOut DDI split (CellOut DDI frame is identical); independent fresh '
                                'seed42 same-DDI-split replicates, NOT true cell-cold DDI generalization')
    report['passed'] = bool(
        all(report['source_pairout_vs_cellout'].values()) and all(report['prepared_pairout_vs_cellout'].values())
        and report['pair_indices_pairout_equal_cellout'] and report['event_names_exact']
        and all(all(a['prepared_vs_source'].values()) and a['counts_exact'] and a['classes_exact']
                and a['pair_indices_rows_exact'] and a['all_drugs_have_graph'] and a['all_drugs_have_nai_row']
                and a['event_ids_in_0_84'] for a in report['arms'].values())
        and report['schedule'] == {'train_batches': 1184, 'planned_updates': 177600, 'kl_warmup_updates': 35520,
                                   'kl_warmup_epochs': 30.}
        and all(report['copy'][k] for k in ['model_py_byte_identical_to_accelerated_four',
                                            'data_py_byte_identical_to_accelerated_four',
                                            'accelerated_four_byte_identical_to_gpu3_verified',
                                            'prior_gpu3_passed', 'prior_approved'])
        and all(report['inventory_checks'].values()))
    output = CONTROL / 'evidence/cohort_copy_evidence.json'
    output.write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
    print(json.dumps({'passed': report['passed'], 'arms': report['arms'], 'schedule': report['schedule'],
                      'copy': report['copy'], 'inventory_groups': inventory['groups'],
                      'inventory_checks': report['inventory_checks']}, indent=1))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

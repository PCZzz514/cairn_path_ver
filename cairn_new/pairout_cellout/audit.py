"""Full real-cache/source audit and syntax review; no model forward or training.

pairout_cellout adaptation of cairn_dsn/audit.py: the same every-pair topology
checks (source facts, multiplicity, inverses, explicit self relations, exact
both-direction depth-4 reachability == mask) on the new cohort cache, plus exact
repack of every reused data/prepared segment, frozen PairOut/CellOut rows, event
names and global drug IDs, split leakage, molecular graphs, FIN/NAI tables and
the two full-path configurations. Writes pairout_cellout/prepared/audit_report.json.
"""
import ast
import filecmp
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from .dsn.train import SCHEDULE, TRAIN_BATCHES
from .prepare import COLUMNS, EXPECTED, OLD, SOURCES, file_stats, leakage

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'pairout_cellout/prepared'
DEV = ROOT.parent / 'cairn_dev'
GPU = {'pairout': 6, 'cellout': 7}
UNCHANGED = {'pairout_cellout/dsn/model.py': 'accelerated_four/dsn/model.py',
             'pairout_cellout/dsn/data.py': 'accelerated_four/dsn/data.py',
             'pairout_cellout/dsn/metrics.py': 'accelerated_four/dsn/metrics.py',
             'pairout_cellout/paths.py': 'cairn_dsn/paths.py'}


def main():
    syntax = []
    for file in sorted((ROOT / 'pairout_cellout').rglob('*.py')):
        ast.parse(file.read_text(), filename=str(file))
        syntax.append(str(file.relative_to(ROOT)))
    for copy, original in UNCHANGED.items():
        assert filecmp.cmp(ROOT / copy, ROOT / original, shallow=False), f'{copy} differs from {original}'
    preparation = json.loads((DATA / 'preparation_report.json').read_text())
    assert preparation['namespace'] == 'pairout_cellout'
    assert file_stats(OLD) == preparation['original_prepared_file_stats'], 'data/prepared changed'
    assets = torch.load(DATA / 'assets.pt', map_location='cpu', weights_only=False)
    n_rel, n_ent = len(assets['fin_relation_ids']), len(assets['fin_entity_ids'])
    source = torch.load(DEV / 'data/processed/drkg_triples.pt', map_location='cpu', weights_only=False)
    assert len(source['triples']) == 4488504
    assert source['entities'] == assets['fin_entity_ids'] and source['relations'] == assets['fin_relation_ids']
    h, r, t = source['triples'].numpy().T
    source_types = np.asarray([e.split('::', 1)[0] for e in source['entities']])
    # No DDI (Compound-Compound) fact exists in the path background.
    assert not np.any((source_types[h] == 'Compound') & (source_types[t] == 'Compound'))
    source_key, source_count = np.unique((h * n_ent + t) * n_rel + r, return_counts=True)
    arrays = {name: np.load(DATA / f'{name}.npy', mmap_mode='r') for name in
              ['pairs', 'nodes', 'edges', 'node_ptr', 'edge_ptr', 'endpoints', 'path_mask']}
    pairs, nodes, edges = arrays['pairs'], arrays['nodes'], arrays['edges']
    nptr, eptr, ends, masks = [arrays[k] for k in ['node_ptr', 'edge_ptr', 'endpoints', 'path_mask']]
    assert len(nptr) == len(eptr) == len(pairs) + 1
    assert nptr[-1] == len(nodes) and eptr[-1] == len(edges)
    assert np.all(np.diff(nptr) >= 1) and np.all(np.diff(eptr) >= 0)
    real_edges, self_edges, gene_gene, gene_pathway = 0, 0, 0, 0
    type_names = [e.split('::', 1)[0] for e in assets['fin_entity_ids']]
    gene = np.asarray([e == 'Gene' for e in type_names])
    pathway = np.asarray([e == 'Pathway' for e in type_names])
    for first in range(0, len(pairs), 5000):
        last = min(first + 5000, len(pairs))
        local_edges = np.asarray(edges[eptr[first]:eptr[last]], dtype=np.int64)
        edge_pair = np.repeat(np.arange(first, last), np.diff(eptr[first:last + 1]))
        node_pair = np.repeat(np.arange(first, last), np.diff(nptr[first:last + 1]))
        local_nodes = np.asarray(nodes[nptr[first]:nptr[last]], dtype=np.int64)
        count_per_edge = np.diff(nptr)[edge_pair]
        assert np.all(local_edges[:, :2] >= 0) and np.all(local_edges[:, :2] < count_per_edge[:, None])
        assert np.all(local_edges[:, 2] >= 0) and np.all(local_edges[:, 2] <= 2 * n_rel)
        assert np.all(local_edges[:, 3] > 0)
        offset = nptr[edge_pair] - nptr[first]
        src_idx, dst_idx = local_edges[:, 0] + offset, local_edges[:, 1] + offset
        src, dst = local_nodes[src_idx], local_nodes[dst_idx]
        rel = local_edges[:, 2]
        real = rel < 2 * n_rel
        forward = rel[real] < n_rel
        hs = np.where(forward, src[real], dst[real])
        ts = np.where(forward, dst[real], src[real])
        rs = rel[real] % n_rel
        assert np.all(hs >= 0) and np.all(ts >= 0)
        keys = (hs * n_ent + ts) * n_rel + rs
        position = np.searchsorted(source_key, keys)
        assert np.all(position < len(source_key))
        assert np.array_equal(source_key[position], keys)
        assert np.array_equal(source_count[position], local_edges[real, 3])
        assert np.all(src_idx[~real] == dst_idx[~real])
        # Every real typed edge has its declared inverse, with the same weight.
        inverse_rel = np.where(rel < n_rel, rel + n_rel, rel - n_rel)
        enc = (src_idx[real] * len(local_nodes) + dst_idx[real]) * (2 * n_rel + 1) + rel[real]
        inv = (dst_idx[real] * len(local_nodes) + src_idx[real]) * (2 * n_rel + 1) + inverse_rel[real]
        assert np.array_equal(np.sort(enc), np.sort(inv))
        # Verify exact cached reachability for BOTH directions across all pairs,
        # independent of floating-point weights and all learned model states.
        node_offsets = nptr[first:last] - nptr[first]
        head_nodes = ends[first:last, 0] + node_offsets
        tail_nodes = ends[first:last, 1] + node_offsets
        assert np.array_equal(local_nodes[head_nodes], assets['drug_fin_idx'].numpy()[pairs[first:last, 0]])
        assert np.array_equal(local_nodes[tail_nodes], assets['drug_fin_idx'].numpy()[pairs[first:last, 1]])
        for starts, targets in [(head_nodes, tail_nodes), (tail_nodes, head_nodes)]:
            reachable = np.zeros(len(local_nodes), dtype=bool)
            reachable[starts] = True
            for _ in range(4):
                reached = np.zeros_like(reachable)
                reached[dst_idx[reachable[src_idx]]] = True
                reachable = reached
            assert np.array_equal(reachable[targets], masks[first:last])
        # Missing-path graphs contain only their endpoint placeholders and no
        # fabricated self edge, so unreachable sources die on the first layer.
        assert np.all(np.diff(eptr)[first:last][~masks[first:last]] == 0)
        # Exactly one explicit self relation per node of a present subgraph.
        self_per_pair = np.bincount(edge_pair[~real] - first, minlength=last - first)
        assert np.array_equal(self_per_pair, np.diff(nptr)[first:last] * masks[first:last])
        real_edges += int(real.sum())
        self_edges += int((~real).sum())
        gene_gene += int((gene[hs] & gene[ts]).sum())
        gene_pathway += int(((gene[hs] & pathway[ts]) | (pathway[hs] & gene[ts])).sum())
    # Every reused pair is an exact copy of its audited data/prepared segment
    # (nodes, edges, endpoints, mask); only its global offsets changed.
    old = {name: np.load(OLD / f'{name}.npy', mmap_mode='r') for name in arrays}
    source_index = np.load(DATA / 'source_pair_index.npy')
    reused = np.flatnonzero(source_index >= 0)
    assert len(source_index) == len(pairs) and len(np.unique(source_index[reused])) == len(reused)
    assert np.array_equal(old['pairs'][source_index[reused]], pairs[reused])
    old_pairs = set(map(tuple, old['pairs'].tolist()))
    assert not any(tuple(p) in old_pairs for p in pairs[source_index < 0].tolist())
    assert np.array_equal(ends[reused], old['endpoints'][source_index[reused]])
    assert np.array_equal(masks[reused], old['path_mask'][source_index[reused]])
    assert np.array_equal(np.diff(nptr)[reused], np.diff(old['node_ptr'])[source_index[reused]])
    assert np.array_equal(np.diff(eptr)[reused], np.diff(old['edge_ptr'])[source_index[reused]])
    breaks = np.flatnonzero((np.diff(reused) != 1) | (np.diff(source_index[reused]) != 1)) + 1
    for run in np.split(reused, breaks):
        first, last = int(run[0]), int(run[-1])
        j0, j1 = int(source_index[first]), int(source_index[last])
        for name, ptr in [('nodes', 'node_ptr'), ('edges', 'edge_ptr')]:
            new_start, old_start, length = int(arrays[ptr][first]), int(old[ptr][j0]), int(old[ptr][j1 + 1] - old[ptr][j0])
            for start in range(0, length, 4000000):
                stop = min(start + 4000000, length)
                assert np.array_equal(arrays[name][new_start + start:new_start + stop], old[name][old_start + start:old_start + stop])
    # Frozen sources: exact rows/columns/order/events/global IDs, both protocols.
    bundle = torch.load(DEV / 'data/processed/bundle.pt', map_location='cpu', weights_only=False)
    drugs = pd.read_parquet(DEV / 'data/processed/drugs.parquet')
    assert drugs.drug_id.tolist() == bundle['drug_ids'] == assets['drug_ids']
    drug_sources = drugs.source_ids.map(json.loads).tolist()
    events = pd.read_csv(DATA / 'ddi_events.csv')
    assert events.equals(pd.read_csv(DEV / 'data/processed/ddi_events.csv'))
    assert events.event_id.tolist() == list(range(85)) and events.template.tolist() == assets['ddi_event_names'] == bundle['ddi_event_names']
    protocols, frames = {}, {}
    for protocol, path in SOURCES.items():
        cached = pd.read_parquet(DATA / protocol / 'ddi.parquet')
        original = pd.read_parquet(path)
        assert original.equals(cached) and original.dtypes.equals(cached.dtypes) and cached.columns.tolist() == COLUMNS
        assert filecmp.cmp(path, DATA / protocol / 'ddi.parquet', shallow=False)
        assert cached.groupby('split').size().to_dict() == EXPECTED
        assert (np.asarray(assets['ddi_event_names'])[cached.event_id] == cached.template.to_numpy()).all()
        for col, db in [('drug_a', 'source_drugbank_a'), ('drug_b', 'source_drugbank_b')]:
            assert all(name in drug_sources[i] for i, name in zip(cached[col].tolist(), cached[db].tolist()))
        pair_ids = np.load(DATA / protocol / 'pair_indices.npy')
        assert np.array_equal(pairs[pair_ids], cached[['drug_a', 'drug_b']].to_numpy())
        split_leakage = leakage(cached)
        assert all(v[k] == 0 for v in split_leakage.values() for k in
                   ['ordered_pairs', 'reversed_ordered_pairs', 'unordered_pairs', 'pair_groups'])
        assert split_leakage == preparation['protocols'][protocol]['leakage']
        frames[protocol] = cached
        support = {split: np.bincount(cached[cached.split.eq(split)].event_id, minlength=85).tolist() for split in EXPECTED}
        protocols[protocol] = {'all_rows_exact': True, 'all_columns_order_dtypes_exact': True, 'byte_identical_copy': True,
                               'event_names_exact': True, 'global_drug_ids_exact': True, 'pair_index_exact': True,
                               'split_pair_leakage_zero': True, 'counts': cached.groupby('split').size().to_dict(),
                               'classes_present': {s: int(np.count_nonzero(v)) for s, v in support.items()},
                               'zero_support_classes': {s: [i for i, c in enumerate(v) if c == 0] for s, v in support.items()},
                               'missing_path_rate': {s: float(1 - masks[pair_ids[cached.split.eq(s).to_numpy()]].mean()) for s in EXPECTED}}
    assert frames['pairout'].equals(frames['cellout'])
    assert np.array_equal(np.load(DATA / 'pairout/pair_indices.npy'), np.load(DATA / 'cellout/pair_indices.npy'))
    cohort = frames['pairout'][['drug_a', 'drug_b']].drop_duplicates().sort_values(['drug_a', 'drug_b']).to_numpy()
    assert np.array_equal(cohort, pairs)
    # Molecular graphs: unchanged bundle graphs for exactly the active global IDs.
    active = np.unique(pairs)
    assert sorted(assets['graphs']) == active.tolist()
    for d, graph in assets['graphs'].items():
        reference = bundle['graphs'][d]
        assert graph.x.shape[1] == 55 and graph.edge_attr.shape[1] == 17
        assert sorted(graph.keys()) == sorted(reference.keys()) and all(torch.equal(graph[k], reference[k]) for k in graph.keys())
    # Global FIN/NAI tables (pretrained FIN; no published embeddings).
    fin = torch.load(DEV / 'data/processed/fin.pt', map_location='cpu', weights_only=False)
    old_assets = torch.load(OLD / 'assets.pt', map_location='cpu', weights_only=False)
    assert torch.equal(assets['fin_entities'], fin['entities']) and torch.equal(assets['fin_relations'], fin['relations'])
    assert torch.equal(assets['drug_fin_idx'], bundle['drug_fin_idx'])
    assert torch.equal(assets['neighbor_entities'], old_assets['neighbor_entities'])
    assert torch.equal(assets['neighbor_relations'], old_assets['neighbor_relations'])
    nai = pd.read_parquet(DATA / 'nai_neighbors.parquet')
    assert nai.equals(pd.read_parquet(OLD / 'nai_neighbors.parquet'))
    assert set(nai.relation_idx) <= set(preparation['nai']['six_relation_ids'].values())
    assert all(assets['fin_entity_ids'][i].startswith('Gene::') for i in nai.fin_idx.unique())
    assert int((assets['neighbor_entities'] >= 0).sum(1).max()) <= 30
    # Two full-path seed42 configurations; derived schedule from the frozen rows.
    # Latest human update: 150 maximum epochs (not 100), patience 15, KL warmup 20%.
    train_batches = math.ceil(EXPECTED['train'] / 128)
    schedule = {'epochs': 150, 'patience': 15, 'kl_warmup_fraction': .2, 'train_batches': train_batches,
                'planned_updates': 150 * train_batches, 'kl_warmup_updates': math.ceil(.2 * 150 * train_batches)}
    assert schedule == {'epochs': 150, 'patience': 15, 'kl_warmup_fraction': .2, 'train_batches': 1184,
                        'planned_updates': 177600, 'kl_warmup_updates': 35520}
    assert schedule == {**SCHEDULE, 'train_batches': TRAIN_BATCHES}
    expected_cfg = {'with_path': True, 'prepared': 'pairout_cellout/prepared', 'seed': 42, 'epochs': 150, 'patience': 15,
                    'batch_size': 128, 'learning_rate': 1e-4, 'weight_decay': 1e-4, 'reconstruction_weight': .1,
                    'kl_weight': .001, 'kl_warmup_fraction': .2, 'path_depth': 4, 'path_meeting_cap': 32,
                    'path_arm_cap': 2, 'nai_cap': 30, 'cpu_threads': 2, 'num_workers': 2}
    configs = {}
    for name, gpu in GPU.items():
        cfg = yaml.safe_load((ROOT / 'pairout_cellout/configs' / f'{name}.yaml').read_text())
        assert cfg['protocol'] == name and cfg['gpu'] == gpu and cfg['output'] == f'pairout_cellout/outputs/{name}'
        assert all(cfg[k] == v for k, v in expected_cfg.items()), name
        # No extra keys (e.g. planned_updates/kl_warmup_updates overrides): the
        # trainer derives the schedule only from these reviewed values.
        assert set(cfg) == set(expected_cfg) | {'protocol', 'gpu', 'output', 'ddi_split_note'}, name
        configs[name] = {'gpu': gpu, 'output': cfg['output']}
    report = {'namespace': 'pairout_cellout', 'syntax': syntax, 'unchanged_copies': UNCHANGED,
              'protocols': protocols, 'pairout_cellout_frames_identical': True,
              'source_identity_note': preparation['source_identity']['note'],
              'all_cached_real_edges_found_in_source': True,
              'source_fact_multiplicity_preserved': True, 'all_real_edges_have_inverse': True,
              'endpoint_mapping_exact': True, 'all_pairs_depth4_reachability_equals_explicit_mask': True,
              'missing_paths_have_no_edges': True, 'explicit_self_count_exact': True,
              'no_compound_compound_background_facts': True,
              'reused_segments_exact_copies': {'pairs': len(reused), 'runs': len(breaks) + 1},
              'newly_extracted_pairs': int((source_index < 0).sum()),
              'molecular_graphs_exact_from_bundle': len(assets['graphs']),
              'fin_tables_exact': True, 'nai_table_equal_to_data_prepared': True,
              'original_data_prepared_unchanged': True,
              'pairs': len(pairs), 'pairs_with_paths': int(masks.sum()), 'missing_paths': int((~masks).sum()),
              'missing_path_rate': float(1 - masks.mean()),
              'real_edges': real_edges, 'explicit_self_edges': self_edges,
              'cached_gene_gene_edges_both_directions': gene_gene,
              'cached_gene_pathway_edges_both_directions': gene_pathway,
              'max_nodes': int(np.diff(nptr).max()), 'max_edges': int(np.diff(eptr).max()),
              'gpu_config': GPU, 'configs': configs, 'schedule': schedule,
              'validation': 'Full real-data and static checks only; no model forward, smoke experiment, or GPU training',
              'passed': True}
    (DATA / 'audit_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

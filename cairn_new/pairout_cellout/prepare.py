"""Prepare the frozen PairOut/CellOut DDI rows and label-free graph topology on CPU.

pairout_cellout adaptation of cairn_dsn/prepare.py. The old data/prepared cache
covers ONLY the single/double DrugOut pair union, so the cohort is rebuilt here:
every ordered (drug_a, drug_b) pair of the frozen frames gets a NEW pair index.
Topology of a pair already in data/prepared is copied verbatim from its audited
segment (it depends only on the two global endpoints); missing pairs are
extracted with the unchanged original RouteExtractor (pairout_cellout/paths.py ==
cairn_dsn/paths.py) on the same full non-DDI DRKG with the same caps. A spaced
sample of reused pairs is re-extracted and must equal the old segments exactly.
Global drug IDs, FIN tables, the six-type NAI table and molecular graphs keep the
original global indexing. Read-only inputs: ../cairn_dev, data/prepared.
"""
import argparse
from collections import Counter
import filecmp
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from .paths import RouteExtractor

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT.parent / 'cairn_dev'
PROCESSED = DEV / 'data/processed'
OLD = ROOT / 'data/prepared'
OUT = ROOT / 'pairout_cellout/prepared'
# PairOut is the stage-1 processed frame (outputs/stage1/config.yaml processed:
# data/processed); there is no reference85/pairout directory. CellOut keeps the
# PairOut DDI split (its ddi.parquet links to the same processed frame).
SOURCES = {'pairout': PROCESSED / 'ddi.parquet',
           'cellout': DEV / 'data/experiments/reference85/cellout/ddi.parquet'}
EXPECTED = {'train': 151508, 'val': 18939, 'test': 18938}
COLUMNS = ['drug_a', 'drug_b', 'event_id', 'split', 'source_drugbank_a', 'source_drugbank_b',
           'source_y', 'source_template', 'template', 'pair_group']
TOPOLOGY = ['nodes', 'edges', 'node_ptr', 'edge_ptr', 'endpoints', 'path_mask']
SAMPLE_REEXTRACT = 2048
EXTRACTOR = None
DRUG_FIN = None


def extract_job(job):
    a, b = job
    h, t = int(DRUG_FIN[a]), int(DRUG_FIN[b])
    nodes, edges, present = EXTRACTOR.extract(h, t)
    ends = [int(np.where(nodes == h)[0][0]), int(np.where(nodes == t)[0][-1])]
    return nodes, edges, ends, present


def file_stats(directory):
    return {str(p.relative_to(directory)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(directory.rglob('*')) if p.is_file()}


def pair_sets(frame, split):
    chosen = frame[frame.split.eq(split)]
    ordered = set(chosen[['drug_a', 'drug_b']].itertuples(index=False, name=None))
    return ordered, {tuple(sorted(p)) for p in ordered}, set(chosen.pair_group), set(np.unique(chosen[['drug_a', 'drug_b']]))


def leakage(frame):
    sets = {split: pair_sets(frame, split) for split in EXPECTED}
    result = {}
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        lo, lu, lg, ld = sets[left]
        ro, ru, rg, rd = sets[right]
        result[f'{left}_{right}'] = {'ordered_pairs': len(lo & ro), 'reversed_ordered_pairs': len(lo & {(b, a) for a, b in ro}),
                                     'unordered_pairs': len(lu & ru), 'pair_groups': len(lg & rg),
                                     'shared_drugs_expected_for_pairout': len(ld & rd)}
    return result


def main():
    global EXTRACTOR, DRUG_FIN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--meeting-cap', type=int, default=32)
    parser.add_argument('--arm-cap', type=int, default=2)
    args = parser.parse_args()
    assert (args.meeting_cap, args.arm_cap) == (32, 2), 'same bounded construction as data/prepared'
    OUT.mkdir(parents=True, exist_ok=True)
    original_stats = file_stats(OLD)
    old_report = json.loads((OLD / 'preparation_report.json').read_text())
    old_audit = json.loads((OLD / 'audit_report.json').read_text())
    assert (old_report['paths']['depth'], old_report['paths']['meeting_cap'], old_report['paths']['arm_cap']) == (4, 32, 2)
    assert old_audit['all_cached_real_edges_found_in_source'] and old_audit['all_pairs_depth4_reachability_equals_explicit_mask']
    bundle = torch.load(PROCESSED / 'bundle.pt', map_location='cpu', weights_only=False)
    fin = torch.load(PROCESSED / 'fin.pt', map_location='cpu', weights_only=False)
    drkg = torch.load(PROCESSED / 'drkg_triples.pt', map_location='cpu', weights_only=False)
    old_assets = torch.load(OLD / 'assets.pt', map_location='cpu', weights_only=False)
    entities, relations = drkg['entities'], drkg['relations']
    assert fin['entity_ids'] == entities and fin['relation_ids'] == relations
    assert pd.read_csv(PROCESSED / 'drkg_entities.csv').entity.tolist() == entities
    assert pd.read_csv(PROCESSED / 'drkg_relations.csv').relation.tolist() == relations
    assert torch.equal(fin['entities'], bundle['fin_entities'])
    assert torch.equal(fin['relations'], bundle['fin_relations'])
    # Reuse the global FIN tables and drug->FIN mapping (no renumbering).
    assert old_assets['drug_ids'] == bundle['drug_ids'] and torch.equal(old_assets['drug_fin_idx'], bundle['drug_fin_idx'])
    assert torch.equal(old_assets['fin_entities'], fin['entities']) and torch.equal(old_assets['fin_relations'], fin['relations'])
    assert old_assets['fin_entity_ids'] == entities and old_assets['fin_relation_ids'] == relations
    events = pd.read_csv(PROCESSED / 'ddi_events.csv')
    assert events.event_id.tolist() == list(range(85))
    assert events.template.tolist() == bundle['ddi_event_names'] == old_assets['ddi_event_names']
    assert events.equals(pd.read_csv(OLD / 'ddi_events.csv'))
    drugs = pd.read_parquet(PROCESSED / 'drugs.parquet')
    assert drugs.drug_id.tolist() == bundle['drug_ids']
    drug_sources = drugs.source_ids.map(json.loads).tolist()
    frames, report = {}, {'namespace': 'pairout_cellout', 'protocols': {}}
    for protocol, source in SOURCES.items():
        frame = pd.read_parquet(source)
        counts = frame.groupby('split').size().to_dict()
        assert counts == EXPECTED and frame.columns.tolist() == COLUMNS
        assert frame.event_id.between(0, 84).all()
        for col in ['drug_a', 'drug_b']:
            assert frame[col].between(0, len(bundle['drug_ids']) - 1).all()
        # Exact original event names and global drug identities on every row.
        assert (np.asarray(bundle['ddi_event_names'])[frame.event_id] == frame.template.to_numpy()).all()
        for col, db in [('drug_a', 'source_drugbank_a'), ('drug_b', 'source_drugbank_b')]:
            assert all(name in drug_sources[i] for i, name in zip(frame[col].tolist(), frame[db].tolist()))
        assert not frame.duplicated(['drug_a', 'drug_b']).any() and not (frame.drug_a == frame.drug_b).any()
        dest = OUT / protocol
        dest.mkdir(exist_ok=True)
        shutil.copyfile(source, dest / 'ddi.parquet')
        cached = pd.read_parquet(dest / 'ddi.parquet')
        assert cached.equals(frame) and cached.dtypes.equals(frame.dtypes)
        assert filecmp.cmp(source, dest / 'ddi.parquet', shallow=False)
        frames[protocol] = frame
        report['protocols'][protocol] = {'counts': counts, 'source': str(source), 'resolved_source': os.path.realpath(source),
                                          'exact_dataframe_equal': True, 'byte_identical_copy': True,
                                          'leakage': leakage(frame), 'splits': {}}
    identical = frames['pairout'].equals(frames['cellout'])
    assert identical, 'PairOut and CellOut DDI frames must be exactly identical'
    report['source_identity'] = {
        'pairout_cellout_dataframes_identical': True,
        'byte_identical_sources': filecmp.cmp(SOURCES['pairout'], SOURCES['cellout'], shallow=False),
        'resolved_sources': {p: os.path.realpath(s) for p, s in SOURCES.items()},
        'note': ('CellOut DDI rows, order, labels and split are exactly the PairOut frame; the cell split only '
                 'affects cell-level tasks. Both DDI-only jobs are independent fresh seed42 same-DDI-split '
                 'replicates, NOT DDI cell-cold generalization. No synergy/LINCS/DepMap/cell features are used.')}
    shutil.copyfile(PROCESSED / 'ddi_events.csv', OUT / 'ddi_events.csv')
    frame = frames['pairout']
    pairs = frame[['drug_a', 'drug_b']].drop_duplicates().sort_values(['drug_a', 'drug_b'])
    pairs = pairs.to_numpy(np.int32)
    np.save(OUT / 'pairs.npy', pairs)
    active = np.unique(pairs)
    assert all(bundle['graphs'][int(d)].x.shape[1] == 55 and
               bundle['graphs'][int(d)].edge_attr.shape[1] == 17 for d in active)
    # Original IDs stay global; store only active molecular graphs, unchanged
    # from the read-only bundle. Graphs already in data/prepared must be equal.
    graphs = {int(d): bundle['graphs'][int(d)] for d in active}
    reused_graphs = [d for d in graphs if d in old_assets['graphs']]
    for d in reused_graphs:
        old, new = old_assets['graphs'][d], graphs[d]
        assert sorted(old.keys()) == sorted(new.keys()) and all(torch.equal(old[k], new[k]) for k in new.keys())
    report['graphs'] = {'active_drugs': len(active), 'reused_identical_to_data_prepared': len(reused_graphs),
                        'added_from_bundle': len(graphs) - len(reused_graphs), 'atom_features': 55, 'bond_features': 17}
    # Reuse the existing audited six-type neighborhood table (all DrugBank
    # aliases + original XML target prefix). Re-derive the deterministic balanced
    # cap with the original selection; it must equal the global data/prepared table.
    source_neigh = DEV / 'data/experiments/no_gene_embeddings_six_relations/shared/neighbours.parquet'
    neighbors = pd.read_parquet(source_neigh)
    six_file = DEV / 'references/RareDDIE_source/RareDDIE-f658cae053af1e93eecc88ad77dc3af2e33cf527/RareDDIE/dataset1/dti_rel.csv'
    six = pd.read_csv(six_file, header=None).iloc[:, 0].tolist()
    six_ids = [relations.index(name) for name in six]
    assert set(neighbors.relation_idx.unique()) == set(six_ids)
    assert all(entities[i].startswith('Gene::') for i in neighbors.fin_idx.unique())
    assert all(bundle['drug_ids'][int(d)] == drug for d, drug in
               neighbors[['drug', 'drug_id']].drop_duplicates().itertuples(index=False, name=None))
    local_ent = torch.full((len(bundle['drug_ids']), 30), -1, dtype=torch.long)
    local_rel = torch.full_like(local_ent, -1)
    selected_rows = []
    for drug, group in neighbors.groupby('drug', sort=False):
        # Original slot order within each relation, round-robin in official
        # dti_rel.csv relation order. Cap never changes DRKG or FIN indices.
        buckets = [group[group.relation_idx == r].sort_values('slot').to_dict('records') for r in six_ids]
        chosen = []
        for position in range(30):
            for bucket in buckets:
                if position < len(bucket):
                    chosen.append(bucket[position])
                    if len(chosen) == 30:
                        break
            if len(chosen) == 30:
                break
        for slot, row in enumerate(chosen):
            local_ent[int(drug), slot] = int(row['fin_idx'])
            local_rel[int(drug), slot] = int(row['relation_idx'])
            selected_rows.append({**row, 'selected_slot': slot})
    assert torch.equal(local_ent, old_assets['neighbor_entities']) and torch.equal(local_rel, old_assets['neighbor_relations'])
    selected_rows = pd.DataFrame(selected_rows)
    assert selected_rows.equals(pd.read_parquet(OLD / 'nai_neighbors.parquet'))
    selected_rows.to_parquet(OUT / 'nai_neighbors.parquet', index=False)
    DRUG_FIN = bundle['drug_fin_idx'].numpy()
    torch.save({'graphs': graphs, 'drug_ids': bundle['drug_ids'], 'drug_fin_idx': bundle['drug_fin_idx'],
                'fin_entities': fin['entities'], 'fin_relations': fin['relations'],
                'fin_entity_ids': entities, 'fin_relation_ids': relations,
                'neighbor_entities': local_ent, 'neighbor_relations': local_rel,
                'ddi_event_names': bundle['ddi_event_names'], 'num_classes': 85}, OUT / 'assets.pt')
    report['nai'] = {'source': str(source_neigh), 'cap': 30, 'selection': 'official relation order round-robin; original slot order within relation',
                     'equal_to_data_prepared_global_table': True,
                     'six_relation_ids': dict(zip(six, six_ids)), 'selected_per_relation': {relations[r]: int((local_rel == r).sum()) for r in six_ids},
                     'active_drugs': len(active), 'active_with_neighbors': int((local_ent[active] >= 0).any(1).sum()),
                     'active_truncated': int(neighbors.groupby('drug').size().reindex(active, fill_value=0).gt(30).sum())}
    triples = drkg['triples'].numpy()
    types = np.asarray([e.split('::', 1)[0] for e in entities])
    assert not np.any((types[triples[:, 0]] == 'Compound') & (types[triples[:, 2]] == 'Compound'))
    relation_counts = np.bincount(triples[:, 1], minlength=len(relations))
    report['drkg'] = {'triples': len(triples), 'entities': len(entities), 'relations': len(relations),
                      'compound_compound': 0, 'gene_gene': int(((types[triples[:, 0]] == 'Gene') & (types[triples[:, 2]] == 'Gene')).sum()),
                      'gene_pathway': int(((types[triples[:, 0]] == 'Gene') & (types[triples[:, 2]] == 'Pathway')).sum() +
                                          ((types[triples[:, 2]] == 'Gene') & (types[triples[:, 0]] == 'Pathway')).sum()),
                      'relation_counts': dict(zip(relations, relation_counts.tolist()))}
    assert report['drkg']['triples'] == old_report['drkg']['triples'] == 4488504
    # Map every new ordered pair to its audited data/prepared segment, if any.
    old = {name: np.load(OLD / f'{name}.npy', mmap_mode='r') for name in ['pairs', *TOPOLOGY]}
    old_index = {tuple(p): i for i, p in enumerate(old['pairs'].tolist())}
    source_index = np.asarray([old_index.get(tuple(p), -1) for p in pairs.tolist()], np.int64)
    missing = np.flatnonzero(source_index < 0)
    reused = np.flatnonzero(source_index >= 0)
    sample = reused[np.unique(np.linspace(0, len(reused) - 1, SAMPLE_REEXTRACT).astype(np.int64))]
    print(f'{len(pairs)} cohort pairs: {len(reused)} reused, {len(missing)} to extract, '
          f'{len(sample)} reused re-extracted for equality', flush=True)
    print('Building full-graph two-hop endpoint balls', flush=True)
    EXTRACTOR = RouteExtractor(triples, entities, len(relations), args.meeting_cap, args.arm_cap)
    jobs = np.concatenate([missing, sample])
    endpoints = np.unique(DRUG_FIN[pairs[jobs]])
    EXTRACTOR.build_balls(endpoints[endpoints >= 0])
    print(f'Extracting {len(jobs)} full-data pair subgraphs', flush=True)
    extracted = {}
    with mp.get_context('fork').Pool(args.workers) as pool:
        for i, result in zip(jobs.tolist(), pool.imap(extract_job, pairs[jobs].tolist(), chunksize=32)):
            extracted[i] = result
    sample_mismatch = []
    for i in sample.tolist():
        nodes, edges, ends, present = extracted.pop(i)
        j = int(source_index[i])
        same = (np.array_equal(nodes, old['nodes'][old['node_ptr'][j]:old['node_ptr'][j + 1]]) and
                np.array_equal(edges.reshape(-1, 4), old['edges'][old['edge_ptr'][j]:old['edge_ptr'][j + 1]]) and
                np.array_equal(ends, old['endpoints'][j]) and bool(present) == bool(old['path_mask'][j]))
        if not same:
            sample_mismatch.append(i)
    assert not sample_mismatch, f'Re-extraction differs from data/prepared for cohort pairs {sample_mismatch[:10]}'
    node_counts = np.zeros(len(pairs), np.int64)
    edge_counts = np.zeros(len(pairs), np.int64)
    node_counts[reused] = np.diff(old['node_ptr'])[source_index[reused]]
    edge_counts[reused] = np.diff(old['edge_ptr'])[source_index[reused]]
    for i, (nodes, edges, ends, present) in extracted.items():
        node_counts[i], edge_counts[i] = len(nodes), len(edges)
    node_ptr = np.concatenate([[0], node_counts.cumsum()]).astype(np.int64)
    edge_ptr = np.concatenate([[0], edge_counts.cumsum()]).astype(np.int64)
    endpoint_rows = np.zeros((len(pairs), 2), np.int64)
    masks = np.zeros(len(pairs), bool)
    endpoint_rows[reused] = old['endpoints'][source_index[reused]]
    masks[reused] = old['path_mask'][source_index[reused]]
    nodes_out = np.lib.format.open_memmap(OUT / 'nodes.npy', mode='w+', dtype=np.int32, shape=(int(node_ptr[-1]),))
    edges_out = np.lib.format.open_memmap(OUT / 'edges.npy', mode='w+', dtype=np.int32, shape=(int(edge_ptr[-1]), 4))
    # Copy contiguous runs of reused pairs (consecutive new AND old indices) as
    # whole old segments; local node indices are pair-relative, so only the
    # global offsets (pointers) change.
    breaks = np.flatnonzero((np.diff(reused) != 1) | (np.diff(source_index[reused]) != 1)) + 1
    for run in np.split(reused, breaks):
        first, last = int(run[0]), int(run[-1])
        j0, j1 = int(source_index[first]), int(source_index[last])
        for name, ptr, target in [('nodes', node_ptr, nodes_out), ('edges', edge_ptr, edges_out)]:
            source_ptr = old[f'{name[:4]}_ptr']
            step = 4000000
            for start in range(int(source_ptr[j0]), int(source_ptr[j1 + 1]), step):
                stop = min(start + step, int(source_ptr[j1 + 1]))
                offset = int(ptr[first]) + start - int(source_ptr[j0])
                target[offset:offset + stop - start] = old[name][start:stop]
    for i, (nodes, edges, ends, present) in extracted.items():
        nodes_out[node_ptr[i]:node_ptr[i + 1]] = nodes
        edges_out[edge_ptr[i]:edge_ptr[i + 1]] = edges.reshape(-1, 4)
        endpoint_rows[i], masks[i] = ends, present
    retained = Counter()
    for start in range(0, len(edges_out), 4000000):
        retained.update(dict(enumerate(np.bincount(edges_out[start:start + 4000000, 2], minlength=2 * len(relations) + 1).tolist())))
    edges_out.flush()
    nodes_out.flush()
    del nodes_out, edges_out
    for name, array in [('node_ptr', node_ptr), ('edge_ptr', edge_ptr), ('endpoints', endpoint_rows), ('path_mask', masks)]:
        np.save(OUT / f'{name}.npy', np.asarray(array, dtype=bool if name == 'path_mask' else np.int64))
    np.save(OUT / 'source_pair_index.npy', source_index)
    pair_index = {tuple(p): i for i, p in enumerate(pairs.tolist())}
    support_rows = []
    is_new = source_index < 0
    for protocol, frame in frames.items():
        indices = np.asarray([pair_index[a, b] for a, b in frame[['drug_a', 'drug_b']].itertuples(index=False, name=None)], np.int64)
        np.save(OUT / protocol / 'pair_indices.npy', indices)
        for split in ['train', 'val', 'test']:
            chosen = frame.split.eq(split).to_numpy()
            rows = indices[chosen]
            split_drugs = np.unique(frame.loc[chosen, ['drug_a', 'drug_b']].to_numpy())
            support = np.bincount(frame.loc[chosen, 'event_id'], minlength=85)
            support_rows += [{'protocol': protocol, 'split': split, 'event_id': e, 'support': int(s)} for e, s in enumerate(support)]
            report['protocols'][protocol]['splits'][split] = {
                'rows': int(chosen.sum()), 'classes_present': int((support > 0).sum()),
                'zero_support_classes': np.flatnonzero(support == 0).tolist(),
                'min_support_present': int(support[support > 0].min()), 'max_support': int(support.max()),
                'with_real_path_le4': int(masks[rows].sum()),
                'path_fraction': float(masks[rows].mean()), 'missing_path_rate': float(1 - masks[rows].mean()),
                'rows_reusing_data_prepared_topology': int((~is_new[rows]).sum()),
                'rows_with_newly_extracted_topology': int(is_new[rows].sum()),
                'missing_path_rate_new_pairs': float(1 - masks[rows[is_new[rows]]].mean()) if is_new[rows].any() else None,
                'mapped_drugs': int((DRUG_FIN[split_drugs] >= 0).sum()), 'drugs': len(split_drugs),
                'nai_drugs_present': int((local_ent[split_drugs] >= 0).any(1).sum())}
    pd.DataFrame(support_rows).to_csv(OUT / 'class_support.csv', index=False)
    report['paths'] = {'unique_ordered_pairs': len(pairs), 'depth': 4, 'meeting_cap': args.meeting_cap,
                       'arm_cap': args.arm_cap, 'nodes': int(node_ptr[-1]), 'edges_with_inverses_self': int(edge_ptr[-1]),
                       'max_nodes': int(node_counts.max()), 'max_edges': int(edge_counts.max()),
                       'pairs_with_paths': int(masks.sum()), 'missing': int((~masks).sum()),
                       'missing_path_rate': float(1 - masks.mean()),
                       'reused_from_data_prepared': len(reused), 'newly_extracted': len(missing),
                       'newly_extracted_with_paths': int(masks[missing].sum()),
                       'newly_extracted_missing_path_rate': float(1 - masks[missing].mean()) if len(missing) else None,
                       'data_prepared_pairs_not_in_cohort': int(len(old['pairs']) - len(reused)),
                       'reused_reextracted_equal': {'pairs': len(sample), 'mismatches': 0,
                                                     'selection': f'{SAMPLE_REEXTRACT} evenly spaced reused cohort positions'},
                       'reuse_rule': 'ordered global (drug_a, drug_b) pair; topology depends only on its two FIN endpoints',
                       'source_pair_index': 'source_pair_index.npy: data/prepared pair index or -1 (newly extracted)',
                       'coverage_test': 'full unbounded two-hop-ball intersection before selecting real routes',
                       'label_free': 'endpoints only; no labels, splits, learned states or NAI slots in extraction',
                       'retained_relation_counts': {str(k): v for k, v in sorted(retained.items()) if v}}
    records = [json.loads(line) for line in (DEV / 'outputs/fin/train.jsonl').read_text().splitlines()]
    report['fin'] = {'selected_epoch': fin['epoch'], 'completed_epoch': records[-1]['epoch'],
                     'dimension': fin['dimension'], 'training_loss': fin['training_loss'],
                     'mapping_order_verified': True, 'published_embeddings_used': False,
                     'equal_to_data_prepared_global_table': True, 'source': str(PROCESSED / 'fin.pt')}
    assert file_stats(OLD) == original_stats, 'data/prepared changed during preparation'
    report['original_prepared_file_stats'] = original_stats
    (OUT / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'complete': True, 'paths': {k: v for k, v in report['paths'].items() if k != 'retained_relation_counts'},
                      'protocols': report['protocols'], 'source_identity': report['source_identity']}, indent=2), flush=True)


if __name__ == '__main__':
    main()

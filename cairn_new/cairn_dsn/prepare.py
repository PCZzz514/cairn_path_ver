"""Prepare all real frozen DDI rows and reusable label-free graph topology on CPU."""
import argparse
from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from .paths import RouteExtractor

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT.parent / 'cairn_dev'
PROCESSED = DEV / 'data/processed'
OUT = ROOT / 'data/prepared'
EXTRACTOR = None
DRUG_FIN = None


def extract_job(job):
    a, b = job
    h, t = int(DRUG_FIN[a]), int(DRUG_FIN[b])
    nodes, edges, present = EXTRACTOR.extract(h, t)
    ends = [int(np.where(nodes == h)[0][0]), int(np.where(nodes == t)[0][-1])]
    return nodes, edges, ends, present


def main():
    global EXTRACTOR, DRUG_FIN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--meeting-cap', type=int, default=32)
    parser.add_argument('--arm-cap', type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(PROCESSED / 'bundle.pt', map_location='cpu', weights_only=False)
    fin = torch.load(PROCESSED / 'fin.pt', map_location='cpu', weights_only=False)
    drkg = torch.load(PROCESSED / 'drkg_triples.pt', map_location='cpu', weights_only=False)
    entities, relations = drkg['entities'], drkg['relations']
    assert fin['entity_ids'] == entities and fin['relation_ids'] == relations
    assert pd.read_csv(PROCESSED / 'drkg_entities.csv').entity.tolist() == entities
    assert pd.read_csv(PROCESSED / 'drkg_relations.csv').relation.tolist() == relations
    assert torch.equal(fin['entities'], bundle['fin_entities'])
    assert torch.equal(fin['relations'], bundle['fin_relations'])
    events = pd.read_csv(PROCESSED / 'ddi_events.csv')
    assert events.event_id.tolist() == list(range(85))
    assert events.template.tolist() == bundle['ddi_event_names']
    frames, report = {}, {'protocols': {}}
    expected = {'single_drugout': {'train': 149048, 'val': 20785, 'test': 16978},
                'double_drugout': {'train': 149048, 'val': 760, 'test': 534}}
    for protocol in expected:
        source = DEV / 'data/experiments/reference85' / protocol / 'ddi.parquet'
        frame = pd.read_parquet(source)
        counts = frame.groupby('split').size().to_dict()
        assert counts == expected[protocol]
        assert frame.event_id.between(0, 84).all()
        for col in ['drug_a', 'drug_b']:
            assert frame[col].between(0, len(bundle['drug_ids']) - 1).all()
        dest = OUT / protocol
        dest.mkdir(exist_ok=True)
        shutil.copyfile(source, dest / 'ddi.parquet')
        assert pd.read_parquet(dest / 'ddi.parquet').equals(frame)
        frames[protocol] = frame
        report['protocols'][protocol] = {'counts': counts, 'source': str(source),
                                          'exact_dataframe_equal': True, 'splits': {}}
    shutil.copyfile(PROCESSED / 'ddi_events.csv', OUT / 'ddi_events.csv')
    pairs = pd.concat([f[['drug_a', 'drug_b']] for f in frames.values()]).drop_duplicates().sort_values(['drug_a', 'drug_b'])
    pairs = pairs.to_numpy(np.int32)
    np.save(OUT / 'pairs.npy', pairs)
    active = np.unique(pairs)
    assert all(bundle['graphs'][int(d)].x.shape[1] == 55 and
               bundle['graphs'][int(d)].edge_attr.shape[1] == 17 for d in active)
    # Original IDs stay global; store only active molecular graphs.
    graphs = {int(d): bundle['graphs'][int(d)] for d in active}
    # Reuse the existing audited six-type neighborhood table (all DrugBank
    # aliases + original XML target prefix). Select a deterministic balanced cap.
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
    pd.DataFrame(selected_rows).to_parquet(OUT / 'nai_neighbors.parquet', index=False)
    DRUG_FIN = bundle['drug_fin_idx'].numpy()
    torch.save({'graphs': graphs, 'drug_ids': bundle['drug_ids'], 'drug_fin_idx': bundle['drug_fin_idx'],
                'fin_entities': fin['entities'], 'fin_relations': fin['relations'],
                'fin_entity_ids': entities, 'fin_relation_ids': relations,
                'neighbor_entities': local_ent, 'neighbor_relations': local_rel,
                'ddi_event_names': bundle['ddi_event_names'], 'num_classes': 85}, OUT / 'assets.pt')
    report['nai'] = {'source': str(source_neigh), 'cap': 30, 'selection': 'official relation order round-robin; original slot order within relation',
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
    print('Building full-graph two-hop endpoint balls', flush=True)
    EXTRACTOR = RouteExtractor(triples, entities, len(relations), args.meeting_cap, args.arm_cap)
    endpoints = np.unique(DRUG_FIN[active])
    EXTRACTOR.build_balls(endpoints[endpoints >= 0])
    print(f'Extracting {len(pairs)} full-data pair subgraphs', flush=True)
    # Stream results to workspace binary files, then materialize mmap-friendly
    # .npy arrays. This avoids retaining every subgraph in Python objects.
    node_ptr, edge_ptr, endpoint_rows, masks = [0], [0], [], []
    retained = Counter()
    with (OUT / 'nodes.bin').open('wb') as nf, (OUT / 'edges.bin').open('wb') as ef:
        with mp.get_context('fork').Pool(args.workers) as pool:
            for i, (nodes, edges, ends, present) in enumerate(pool.imap(extract_job, pairs.tolist(), chunksize=128)):
                nodes.tofile(nf)
                edges.tofile(ef)
                node_ptr.append(node_ptr[-1] + len(nodes))
                edge_ptr.append(edge_ptr[-1] + len(edges))
                endpoint_rows.append(ends)
                masks.append(present)
                retained.update(edges[:, 2].tolist())
                if (i + 1) % 10000 == 0:
                    print(f'prepared {i + 1}/{len(pairs)} pairs, {edge_ptr[-1]} edges', flush=True)
    for name, shape in [('nodes', (node_ptr[-1],)), ('edges', (edge_ptr[-1], 4))]:
        raw = np.memmap(OUT / f'{name}.bin', dtype=np.int32, mode='r', shape=shape)
        dest = np.lib.format.open_memmap(OUT / f'{name}.npy', mode='w+', dtype=np.int32, shape=shape)
        for start in range(0, len(raw), 1000000):
            dest[start:start + 1000000] = raw[start:start + 1000000]
        del dest, raw
        (OUT / f'{name}.bin').unlink()
    for name, array in [('node_ptr', node_ptr), ('edge_ptr', edge_ptr), ('endpoints', endpoint_rows), ('path_mask', masks)]:
        np.save(OUT / f'{name}.npy', np.asarray(array, dtype=bool if name == 'path_mask' else np.int64))
    pair_index = {tuple(p): i for i, p in enumerate(pairs.tolist())}
    for protocol, frame in frames.items():
        indices = np.asarray([pair_index[a, b] for a, b in frame[['drug_a', 'drug_b']].itertuples(index=False, name=None)], np.int64)
        np.save(OUT / protocol / 'pair_indices.npy', indices)
        for split in ['train', 'val', 'test']:
            chosen = frame.split.eq(split).to_numpy()
            rows = indices[chosen]
            drugs = np.unique(frame.loc[chosen, ['drug_a', 'drug_b']].to_numpy())
            report['protocols'][protocol]['splits'][split] = {
                'rows': int(chosen.sum()), 'classes_present': int(frame.loc[chosen].event_id.nunique()),
                'with_real_path_le4': int(np.asarray(masks)[rows].sum()),
                'path_fraction': float(np.asarray(masks)[rows].mean()),
                'mapped_drugs': int((DRUG_FIN[drugs] >= 0).sum()), 'drugs': len(drugs),
                'nai_drugs_present': int((local_ent[drugs] >= 0).any(1).sum())}
    report['paths'] = {'unique_ordered_pairs': len(pairs), 'depth': 4, 'meeting_cap': args.meeting_cap,
                       'arm_cap': args.arm_cap, 'nodes': node_ptr[-1], 'edges_with_inverses_self': edge_ptr[-1],
                       'max_nodes': int(np.diff(node_ptr).max()), 'max_edges': int(np.diff(edge_ptr).max()),
                       'pairs_with_paths': int(sum(masks)), 'missing': int(len(masks) - sum(masks)),
                       'coverage_test': 'full unbounded two-hop-ball intersection before selecting real routes',
                       'retained_relation_counts': {str(k): v for k, v in sorted(retained.items())}}
    records = [json.loads(line) for line in (DEV / 'outputs/fin/train.jsonl').read_text().splitlines()]
    report['fin'] = {'selected_epoch': fin['epoch'], 'completed_epoch': records[-1]['epoch'],
                     'dimension': fin['dimension'], 'training_loss': fin['training_loss'],
                     'mapping_order_verified': True, 'published_embeddings_used': False,
                     'source': str(PROCESSED / 'fin.pt')}
    (OUT / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'complete': True, 'paths': report['paths'], 'protocols': report['protocols']}, indent=2), flush=True)


if __name__ == '__main__':
    main()

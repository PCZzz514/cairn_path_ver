"""K-Paths (K=10, L=4) label-free topology cache for every frozen ordered pair.

Reads the read-only DRKG background and the existing frozen pair table
(data/prepared/pairs.npy, the pair_indices of both protocols). Writes the same
array layout the existing Collator reads (nodes/edges/node_ptr/edge_ptr/
endpoints/path_mask) plus per-pair metadata, only under crt_kpaths/cache.
No DDI edge, label, split or old route subgraph enters retrieval.

--sample N measures real per-pair speed on N random pairs (no cache written).
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from .kpaths import Background, Retriever

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT.parent / 'cairn_dev'
PREPARED = ROOT / 'data/prepared'
RETRIEVER = None
PAIRS = None
DRUG_FIN = None


def job(index):
    a, b = (int(x) for x in PAIRS[index])
    nodes, edges, ends, meta = RETRIEVER.retrieve(int(DRUG_FIN[a]), int(DRUG_FIN[b]))
    meta.update(pair_index=int(index), drug_a=a, drug_b=b)
    return nodes, edges, ends, meta


def load_background():
    drkg = torch.load(DEV / 'data/processed/drkg_triples.pt', map_location='cpu', weights_only=False)
    triples, entities, relations = drkg['triples'].numpy(), drkg['entities'], drkg['relations']
    assets = torch.load(PREPARED / 'assets.pt', map_location='cpu', weights_only=False)
    assert assets['fin_entity_ids'] == entities and assets['fin_relation_ids'] == relations
    types = np.asarray([e.split('::', 1)[0] for e in entities])
    # Non-DDI background only: no Compound-Compound fact (so no DDI answer edge).
    assert not np.any((types[triples[:, 0]] == 'Compound') & (types[triples[:, 2]] == 'Compound'))
    return Background(triples, entities, len(relations)), assets['drug_fin_idx'].numpy(), len(triples)


def main():
    global RETRIEVER, PAIRS, DRUG_FIN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--sample', type=int)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--max-length', type=int, default=4)
    parser.add_argument('--budget', type=int, default=1000)
    parser.add_argument('--output', default='crt_kpaths/cache/kpaths_k10_l4')
    args = parser.parse_args()
    started = time.time()
    bg, DRUG_FIN, n_triples = load_background()
    PAIRS = np.load(PREPARED / 'pairs.npy')
    endpoints = np.unique(DRUG_FIN[np.unique(PAIRS)])
    endpoints = endpoints[endpoints >= 0]
    distances = {int(e): bg.distances(int(e)) for e in endpoints}
    # Built once; forked workers share it copy-on-write (never rebuilt per pair).
    RETRIEVER = Retriever(bg, bg.full_graph(), distances, args.k, args.max_length, args.budget)
    setup_seconds = time.time() - started
    print(json.dumps({'phase': 'background', 'directed_pairs': len(bg.pair_src), 'atoms': len(bg.atom_src),
                      'composite_signatures': len(bg.signatures) - 2 * bg.n_rel, 'endpoints': len(endpoints),
                      'seconds': setup_seconds}), flush=True)
    indices = np.arange(len(PAIRS))
    if args.sample:
        indices = np.sort(np.random.default_rng(0).choice(len(PAIRS), args.sample, replace=False))
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    node_ptr, edge_ptr, ends, masks, metas = [0], [0], [], [], []
    run_started = time.time()
    prefix = 'sample_' if args.sample else ''
    with (output / f'{prefix}nodes.bin').open('wb') as nf, (output / f'{prefix}edges.bin').open('wb') as ef:
        with mp.get_context('fork').Pool(args.workers) as pool:
            for i, (nodes, edges, end, meta) in enumerate(pool.imap(job, indices.tolist(), chunksize=4)):
                nodes.tofile(nf)
                edges.tofile(ef)
                node_ptr.append(node_ptr[-1] + len(nodes))
                edge_ptr.append(edge_ptr[-1] + len(edges))
                ends.append(end)
                masks.append(meta['path_mask'])
                metas.append(meta)
                if (i + 1) % 5000 == 0:
                    print(json.dumps({'phase': 'progress', 'pairs': i + 1, 'of': len(indices),
                                      'seconds': time.time() - run_started}), flush=True)
    wall = time.time() - run_started
    frame = pd.DataFrame(metas)
    old_mask = np.load(PREPARED / 'path_mask.npy')[indices]
    frame['old_cache_path_mask'] = old_mask
    seconds = frame.seconds.to_numpy()
    summary = {
        'pairs': len(frame), 'wall_seconds': wall, 'setup_seconds': setup_seconds, 'workers': args.workers,
        'k': args.k, 'max_length': args.max_length, 'candidate_budget': args.budget,
        'graph_budget': {'max_nodes': RETRIEVER.max_nodes, 'max_edges': RETRIEVER.max_edges},
        'pair_seconds': {'mean': float(seconds.mean()), 'p50': float(np.median(seconds)),
                         'p99': float(np.percentile(seconds, 99)), 'max': float(seconds.max()), 'sum': float(seconds.sum())},
        'status_counts': frame.status.value_counts().to_dict(),
        'path_mask_true': int(frame.path_mask.sum()), 'old_cache_path_mask_true': int(old_mask.sum()),
        'budget_truncated': int(frame.budget_truncated.sum()),
        'connected_with_fewer_than_k': int(((frame.status == 'connected_paths') & (frame.retrieved_paths < args.k)).sum()),
        'budget_truncated_with_fewer_than_k': int((frame.budget_truncated & (frame.retrieved_paths < args.k)).sum()),
        'graph_budget_limited': int(frame.graph_budget_reason.notna().sum()),
        'selected_paths_distribution': frame.selected_paths.value_counts().sort_index().to_dict(),
        'shortest_distance_counts': frame.shortest_distance.value_counts().sort_index().to_dict(),
        'max_nodes': int(frame.graph_nodes.max()), 'max_edges': int(frame.graph_edges.max()),
        'mean_nodes': float(frame.graph_nodes.mean()), 'mean_edges': float(frame.graph_edges.mean()),
        'background': {'non_ddi_triples': n_triples, 'directed_node_pairs': len(bg.pair_src),
                       'atomic_directed_relations': len(bg.atom_src), 'relations_with_inverse': 2 * bg.n_rel,
                       'composite_signatures': len(bg.signatures) - 2 * bg.n_rel,
                       'nodes_in_graph': int(bg.in_graph.sum()), 'components': int(bg.component.max() + 1)}}
    if args.sample:
        summary['estimated_full_wall_seconds'] = wall / len(frame) * len(PAIRS)
    for name in ['paths', 'length_counts', 'common_neighbors']:
        frame[name] = frame[name].map(json.dumps)
    frame.to_parquet(output / f'{prefix}metadata.parquet', index=False)
    if not args.sample:
        for name, shape in [('nodes', (node_ptr[-1],)), ('edges', (edge_ptr[-1], 4))]:
            raw = np.memmap(output / f'{name}.bin', dtype=np.int32, mode='r', shape=shape)
            np.save(output / f'{name}.npy', np.asarray(raw))
            del raw
            (output / f'{name}.bin').unlink()
        for name, array, dtype in [('node_ptr', node_ptr, np.int64), ('edge_ptr', edge_ptr, np.int64),
                                   ('endpoints', ends, np.int64), ('path_mask', masks, bool)]:
            np.save(output / f'{name}.npy', np.asarray(array, dtype=dtype))
        # Per-protocol/split coverage, from the frozen pair indices (labels unused).
        summary['protocols'] = {}
        for protocol in ['single_drugout', 'double_drugout']:
            frozen = pd.read_parquet(PREPARED / protocol / 'ddi.parquet')
            pair_ids = np.load(PREPARED / protocol / 'pair_indices.npy')
            summary['protocols'][protocol] = {
                split: {'rows': int(chosen.sum()), 'path_mask_true': int(np.asarray(masks)[pair_ids[chosen]].sum()),
                        'connected_paths': int((frame.status.to_numpy()[pair_ids[chosen]] == 'connected_paths').sum())}
                for split in ['train', 'val', 'test'] for chosen in [frozen.split.eq(split).to_numpy()]}
        summary['complete'] = True
    else:
        for name in ['nodes', 'edges']:
            (output / f'sample_{name}.bin').unlink()
    (output / f'{prefix}report.json').write_text(json.dumps(summary, indent=2, default=int) + '\n')
    print(json.dumps(summary, indent=2, default=int), flush=True)


if __name__ == '__main__':
    main()

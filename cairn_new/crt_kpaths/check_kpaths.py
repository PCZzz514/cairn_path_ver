"""Method-consistency check of crt_kpaths retrieval against the OFFICIAL functions.

Official find_Kpaths / one_node_in_graph / two_nodes_in_graph are loaded by AST
from references/K-Paths (no `datasets` import) and called on the complete
non-DDI DRKG nx.DiGraph (global node-ID insertion; edge 'relation' = the full
sorted signature tuple, independent of our signature-ID table). Real pairs cover
multi-relation hops, repeated relations, direction, one-end-absent, overlength;
no frozen pair is disconnected, so a real drug endpoint is also paired with a
real DRKG node of another component.
Atomic expansion is checked against the raw DRKG triple array. With --cache,
the written cache is checked for 4-layer endpoint reachability == path_mask.
"""
import argparse
import ast
import json
import logging
from pathlib import Path
import time
import networkx as nx
import numpy as np
import pandas as pd
import torch
from .kpaths import Background, Retriever
from .prepare import DEV, PREPARED, ROOT

OFFICIAL = ROOT / 'references/K-Paths/k-paths/utils/Kpaths_utils.py'
OFFICIAL_REVISION = 'bf25d24eac6939574542d8994957b340ea1f5ab8'


def official_functions():
    tree = ast.parse(OFFICIAL.read_text())
    keep = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name in ('find_Kpaths', 'one_node_in_graph', 'two_nodes_in_graph')]
    namespace = {'nx': nx, 'logger': logging.getLogger('official_kpaths')}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(OFFICIAL), 'exec'), namespace)
    return namespace


def official_as_paths(result, signature=list):
    """official [(u, relation, v), ...] -> (node path, signature sequence)."""
    return [([path[0][0]] + [hop[2] for hop in path], [signature(hop[1]) for hop in path]) for path in result]


def ours_as_paths(meta):
    return [(p['nodes'], p['signatures']) for p in meta['paths']]


def full_graph(bg):
    graph = nx.DiGraph()
    graph.add_nodes_from(np.flatnonzero(bg.in_graph).tolist())
    graph.add_edges_from((u, v, {'relation': bg.signatures[s]}) for u, v, s in zip(
        bg.pair_src.tolist(), bg.pair_dst.tolist(), bg.pair_sig.tolist()))
    return graph


def raw_hop_facts(triples, n_rel, u, v):
    forward = triples[(triples[:, 0] == u) & (triples[:, 2] == v), 1]
    backward = triples[(triples[:, 0] == v) & (triples[:, 2] == u), 1] + n_rel
    rels, counts = np.unique(np.concatenate([forward, backward]), return_counts=True)
    return dict(zip(rels.tolist(), counts.tolist()))


def expansion_matches(triples, n_rel, nodes, edges, meta):
    """Cache edges == raw facts of every selected hop in both directions + self loops."""
    expected = {}
    for path in meta['paths']:
        for u, v in zip(path['nodes'][:-1], path['nodes'][1:]):
            for x, y in ((u, v), (v, u)):
                for r, c in raw_hop_facts(triples, n_rel, x, y).items():
                    expected[x, y, r] = c
    got = {(int(nodes[e[0]]), int(nodes[e[1]]), int(e[2])): int(e[3]) for e in edges if e[2] != 2 * n_rel}
    loops = [e for e in edges if e[2] == 2 * n_rel]
    return got == expected and len(loops) == len(nodes) and all(e[0] == e[1] and e[3] == 1 for e in loops)


def reachable_within(edges, head, tail, depth):
    reached = {head}
    for _ in range(depth):
        reached |= {int(v) for u, v, _, _ in edges if int(u) in reached}
    return tail in reached


def check_cache(cache):
    arrays = {n: np.load(cache / f'{n}.npy', mmap_mode='r') for n in
              ['nodes', 'edges', 'node_ptr', 'edge_ptr', 'endpoints', 'path_mask']}
    meta = pd.read_parquet(cache / 'metadata.parquet', columns=['pair_index', 'status', 'path_mask'])
    assert np.array_equal(meta.pair_index, np.arange(len(meta)))
    edge_ptr = np.asarray(arrays['edge_ptr'])
    mismatch = 0
    for i in range(len(meta)):
        edges = np.asarray(arrays['edges'][edge_ptr[i]:edge_ptr[i + 1]])
        head, tail = (int(x) for x in arrays['endpoints'][i])
        # Both directions at EmerGNN depth 4, same as the path branch.
        reach = reachable_within(edges, head, tail, 4) and reachable_within(edges, tail, head, 4)
        mismatch += bool(arrays['path_mask'][i]) != (reach and meta.status[i] == 'connected_paths')
    return {'pairs': len(meta), 'path_mask_equals_connected_depth4_reachability': mismatch == 0,
            'mismatches': mismatch, 'fallback_mask_true': int((meta.path_mask & (meta.status != 'connected_paths')).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-connected', type=int, default=24)
    parser.add_argument('--cache', default=None)
    parser.add_argument('--output', default='crt_kpaths/evidence/kpaths_official_check.json')
    args = parser.parse_args()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.cache:
        report = check_cache(ROOT / args.cache)
        output.with_name('kpaths_cache_check.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return
    official = official_functions()
    drkg = torch.load(DEV / 'data/processed/drkg_triples.pt', map_location='cpu', weights_only=False)
    triples, n_rel = drkg['triples'].numpy(), len(drkg['relations'])
    bg = Background(triples, drkg['entities'], n_rel)
    fin = torch.load(PREPARED / 'assets.pt', map_location='cpu', weights_only=False)['drug_fin_idx'].numpy()
    pairs = fin[np.load(PREPARED / 'pairs.npy')]
    distances = {int(e): bg.distances(int(e)) for e in np.unique(pairs[pairs >= 0])}
    retriever = Retriever(bg, bg.full_graph(), distances)
    started = time.time()
    graph = full_graph(bg)
    build_seconds = time.time() - started
    # Real case selection from the frozen pairs (topology only, no labels).
    rng = np.random.default_rng(1)
    mapped = (pairs >= 0).all(1)
    same = mapped & (bg.component[np.maximum(pairs[:, 0], 0)] == bg.component[np.maximum(pairs[:, 1], 0)])
    cases = {'one_end_absent': np.flatnonzero((pairs >= 0).sum(1) == 1)[:2].tolist(),
             'both_absent': np.flatnonzero((pairs < 0).all(1))[:1].tolist(),
             'disconnected': np.flatnonzero(mapped & ~same)[:3].tolist()}
    # Real nodes of other components paired with a real drug endpoint.
    main_component = bg.component[pairs[mapped][0, 0]]
    others = [int(np.flatnonzero(bg.component == c)[0]) for c in np.unique(bg.component[bg.in_graph])
              if c != main_component][:2]
    synthetic = [(int(pairs[mapped][0, 0]), others[0]), (others[1], int(pairs[mapped][1, 1]))]
    distances.update({node: bg.distances(node) for node in others})  # shared with the retriever
    overlength, connected = [], []
    for i in rng.permutation(np.flatnonzero(same)):
        a, b = pairs[i]
        d = bg.exact_distance(int(a), int(b), distances[int(a)], distances[int(b)])
        if d > 4 and len(overlength) < 3:
            overlength.append(int(i))
        elif d <= 4 and len(connected) < args.random_connected:
            connected.append(int(i))
        if len(overlength) == 3 and len(connected) == args.random_connected:
            break
    cases['overlength'], cases['connected_random'] = overlength, connected
    records, summary = [], {}
    jobs = [(kind, i, *(int(x) for x in pairs[i])) for kind, indices in cases.items() for i in indices]
    jobs += [('disconnected_real_nodes', -1, a, b) for a, b in synthetic]
    cases['disconnected_real_nodes'] = synthetic
    for kind, i, a, b in jobs:
        nodes, edges, ends, meta = retriever.retrieve(a, b)
        t0 = time.time()
        full = official['find_Kpaths'](graph, a, b, max_length=4, max_paths=10)
        official_seconds = time.time() - t0
        record = {'kind': kind, 'pair_index': i, 'fin_a': a, 'fin_b': b, 'status': meta['status'],
                  'path_mask': meta['path_mask'], 'selected': meta['selected_paths'],
                  'candidate_count': meta['candidate_count'], 'is_exact': meta['is_exact'],
                  'ours_seconds': meta['seconds'], 'official_full_graph_seconds': official_seconds,
                  'official_full_graph_paths': len(full),
                  'identical_to_official_full_graph': ours_as_paths(meta) == official_as_paths(full),
                  'same_signature_sequences_as_official_full_graph':
                      sorted(map(str, (s for _, s in ours_as_paths(meta)))) == sorted(map(str, (s for _, s in official_as_paths(full)))),
                  'same_length_profile_as_official_full_graph':
                      [len(p) for p, _ in ours_as_paths(meta)] == [len(p) for p, _ in official_as_paths(full)],
                  'atomic_expansion_matches_raw_triples': expansion_matches(triples, n_rel, nodes, edges, meta)}
        if meta['status'] == 'connected_paths':
            sigs = [s for p in meta['paths'] for s in p['signatures']]
            record['multi_relation_hops'] = sum(len(s) > 1 for s in sigs)
            record['paths_with_repeated_signature'] = sum(
                len({tuple(s) for s in p['signatures']}) < len(p['signatures']) for p in meta['paths'])
            hop = meta['paths'][0]['nodes'][:2]
            forward = graph[hop[0]][hop[1]]['relation']
            backward = graph[hop[1]][hop[0]]['relation']
            record['inverse_signature_reversible'] = backward == tuple(sorted(
                r + n_rel if r < n_rel else r - n_rel for r in forward))
        records.append(record)
        print(json.dumps(record), flush=True)
    connected_records = [r for r in records if r['status'] == 'connected_paths']
    summary = {'official_source': str(OFFICIAL), 'official_revision': OFFICIAL_REVISION,
               'full_graph_nodes': graph.number_of_nodes(), 'full_graph_edges': graph.number_of_edges(),
               'full_graph_build_seconds': build_seconds, 'cases': {k: len(v) for k, v in cases.items()},
               'all_atomic_expansion_match': all(r['atomic_expansion_matches_raw_triples'] for r in records),
               'non_connected_identical_to_official': all(r['identical_to_official_full_graph'] for r in records
                                                          if r['status'] != 'connected_paths'),
               'connected_identical_to_official_full_graph': sum(r['identical_to_official_full_graph'] for r in connected_records),
               'connected_same_signature_sequences_full_graph': sum(r['same_signature_sequences_as_official_full_graph'] for r in connected_records),
               'connected_same_length_profile_full_graph': sum(r['same_length_profile_as_official_full_graph'] for r in connected_records),
               'connected_cases': len(connected_records),
               'multi_relation_hops_seen': sum(r['multi_relation_hops'] for r in connected_records),
               'paths_with_repeated_signature_seen': sum(r['paths_with_repeated_signature'] for r in connected_records),
               'inverse_signature_reversible': all(r['inverse_signature_reversible'] for r in connected_records),
               'fallback_path_mask_false': all(not r['path_mask'] for r in records if r['status'] != 'connected_paths'),
               'records': records}
    output.write_text(json.dumps(summary, indent=2, default=int) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, indent=2, default=int))


if __name__ == '__main__':
    main()

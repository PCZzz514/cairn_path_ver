"""Full real-cache/source audit and syntax review; no model forward or training."""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/prepared'
DEV = ROOT.parent / 'cairn_dev'


def main():
    syntax = []
    for file in sorted((ROOT / 'cairn_dsn').glob('*.py')):
        ast.parse(file.read_text(), filename=str(file))
        syntax.append(str(file.relative_to(ROOT)))
    assets = torch.load(DATA / 'assets.pt', map_location='cpu', weights_only=False)
    n_rel, n_ent = len(assets['fin_relation_ids']), len(assets['fin_entity_ids'])
    source = torch.load(DEV / 'data/processed/drkg_triples.pt', map_location='cpu', weights_only=False)
    h, r, t = source['triples'].numpy().T
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
    protocols = {}
    for protocol in ['single_drugout', 'double_drugout']:
        cached = pd.read_parquet(DATA / protocol / 'ddi.parquet')
        original = pd.read_parquet(DEV / 'data/experiments/reference85' / protocol / 'ddi.parquet')
        assert original.equals(cached)
        pair_ids = np.load(DATA / protocol / 'pair_indices.npy')
        assert np.array_equal(pairs[pair_ids], cached[['drug_a', 'drug_b']].to_numpy())
        protocols[protocol] = {'all_rows_exact': True, 'pair_index_exact': True,
                               'counts': cached.groupby('split').size().to_dict()}
    expected_gpu = {'path_single_drugout': 4, 'path_double_drugout': 5,
                    'no_path_single_drugout': 6, 'no_path_double_drugout': 7}
    for name, gpu in expected_gpu.items():
        cfg = yaml.safe_load((ROOT / 'configs' / f'{name}.yaml').read_text())
        assert cfg['gpu'] == gpu
        assert cfg['epochs'] == 100 and cfg['patience'] == 15 and cfg['batch_size'] == 128
        assert cfg['kl_warmup_fraction'] == .2 and cfg['nai_cap'] == 30 and cfg['path_depth'] == 4
    report = {'syntax': syntax, 'protocols': protocols, 'all_cached_real_edges_found_in_source': True,
              'source_fact_multiplicity_preserved': True, 'all_real_edges_have_inverse': True,
              'endpoint_mapping_exact': True, 'all_pairs_depth4_reachability_equals_explicit_mask': True,
              'missing_paths_have_no_edges': True, 'explicit_self_count_exact': True,
              'real_edges': real_edges, 'explicit_self_edges': self_edges,
              'cached_gene_gene_edges_both_directions': gene_gene,
              'cached_gene_pathway_edges_both_directions': gene_pathway,
              'max_nodes': int(np.diff(nptr).max()), 'max_edges': int(np.diff(eptr).max()),
              'gpu_config': expected_gpu,
              'validation': 'Full real-data and static checks only; no model forward, smoke experiment, or GPU training'}
    (DATA / 'audit_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

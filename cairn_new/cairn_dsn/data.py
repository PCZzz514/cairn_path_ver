"""Frozen-row datasets; raw molecules and packed label-free pair subgraphs."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class DDIDataset(Dataset):
    def __init__(self, prepared, protocol, split):
        directory = Path(prepared) / protocol
        frame = pd.read_parquet(directory / 'ddi.parquet')
        rows = np.flatnonzero(frame.split.eq(split).to_numpy())
        self.frame = frame.iloc[rows].copy()
        self.rows = rows
        self.pair_ids = np.load(directory / 'pair_indices.npy')[rows]
        self.a = self.frame.drug_a.to_numpy(np.int64)
        self.b = self.frame.drug_b.to_numpy(np.int64)
        self.y = self.frame.event_id.to_numpy(np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return int(self.a[i]), int(self.b[i]), int(self.y[i]), int(self.rows[i]), int(self.pair_ids[i])


class Collator:
    def __init__(self, assets, prepared, with_path):
        self.graphs, self.with_path = assets['graphs'], with_path
        if with_path:
            self.paths = {name: np.load(Path(prepared) / f'{name}.npy', mmap_mode='r')
                          for name in ['nodes', 'edges', 'node_ptr', 'edge_ptr', 'endpoints', 'path_mask']}

    def __call__(self, rows):
        a, b, y, original_rows, pairs = zip(*rows)
        ga = [self.graphs[i] for i in a]
        gb = [self.graphs[i] for i in b]
        cross = []
        for h, t in zip(ga, gb):
            # np.meshgrid(x1,x2) ordering in original DSN preprocessing.
            hi, ti = torch.meshgrid(torch.arange(h.num_nodes), torch.arange(t.num_nodes), indexing='xy')
            cross.append(torch.stack([hi.flatten(), ti.flatten()]))
        result = {'graph_a': Batch.from_data_list(ga), 'graph_b': Batch.from_data_list(gb),
                  'cross_edges': cross, 'drug_a': torch.tensor(a), 'drug_b': torch.tensor(b),
                  'label': torch.tensor(y), 'row_index': torch.tensor(original_rows)}
        if self.with_path:
            edge_rows, edge_pairs, head_nodes, tail_nodes, masks = [], [], [], [], []
            offset = 0
            for batch_idx, pair_id in enumerate(pairs):
                start, end = self.paths['edge_ptr'][pair_id:pair_id + 2]
                edges = np.array(self.paths['edges'][start:end], dtype=np.int64)
                edges[:, :2] += offset
                edge_rows.append(edges)
                edge_pairs.append(np.full(len(edges), batch_idx, np.int64))
                endpoints = self.paths['endpoints'][pair_id] + offset
                head_nodes.append(int(endpoints[0]))
                tail_nodes.append(int(endpoints[1]))
                masks.append(bool(self.paths['path_mask'][pair_id]))
                ns, ne = self.paths['node_ptr'][pair_id:pair_id + 2]
                offset += int(ne - ns)
            edges = torch.from_numpy(np.concatenate(edge_rows))
            result['path_graph'] = {'edge_index': edges[:, :2].T.contiguous(), 'relations': edges[:, 2],
                                    'multiplicity': edges[:, 3].float(),
                                    'edge_pair': torch.from_numpy(np.concatenate(edge_pairs)),
                                    'head_nodes': torch.tensor(head_nodes), 'tail_nodes': torch.tensor(tail_nodes),
                                    'path_mask': torch.tensor(masks), 'n_nodes': offset}
        return result


def to_device(value, device):
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    if hasattr(value, 'to'):
        return value.to(device, non_blocking=True)
    return value

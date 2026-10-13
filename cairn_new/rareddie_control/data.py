"""Frozen-row datasets (accelerated_four.dsn.data) and a molecule-only collator.

The accelerated_four Collator always builds DSN cross_edges; this one builds
none and has no path branch.
"""
import torch
from torch_geometric.data import Batch
from accelerated_four.dsn.data import DDIDataset, to_device  # noqa: F401


class Collator:
    def __init__(self, assets):
        self.graphs = assets['graphs']

    def __call__(self, rows):
        a, b, y, original_rows, _ = zip(*rows)
        return {'graph_a': Batch.from_data_list([self.graphs[i] for i in a]),
                'graph_b': Batch.from_data_list([self.graphs[i] for i in b]),
                'drug_a': torch.tensor(a), 'drug_b': torch.tensor(b),
                'label': torch.tensor(y), 'row_index': torch.tensor(original_rows)}

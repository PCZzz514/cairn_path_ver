"""RareDDIE-only control of the accelerated_four no-path model.

Identical to accelerated_four.dsn.model.DSNRareDDI(with_path=False) except the
CSE: the transplanted DSN InterGraphAttention and its 192->128
inter_projection are removed entirely (no modules, no parameters, no cross
edges). Each block is therefore the original RareDDIE MVN_DDI_Block order
(references/RareDDIE/models_t_struc.py): ne_update, feature_conv2, lin_up2,
GlobalPool, normalize/ELU, norm2/ELU. The 128-d atom states from feature_conv2
enter SAGPooling untouched. The molecular TransformerConv attention is kept.
NAI, PVR and the classifier are the validated accelerated_four modules.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, SAGPooling, LayerNorm, Linear, global_add_pool
from accelerated_four.dsn.model import NAI, PairVAE


class CSEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # RareDDIE MVN_DDI_Block: two 2x64 TransformerConvs, bond64. re_shape is
        # never reached by the original forward and is not built.
        self.feature_conv = TransformerConv(128, 64, 2, edge_dim=64, dropout=0.)
        self.lin_up = Linear(64, 64, bias=True, weight_initializer='glorot')
        self.feature_conv2 = TransformerConv(128, 64, 2, edge_dim=64, dropout=0.)
        self.lin_up2 = Linear(64, 64, bias=True, weight_initializer='glorot')
        self.readout = SAGPooling(128, min_score=-1)
        self.norm = LayerNorm(128)
        self.norm2 = LayerNorm(128)
        self.re_shape_e = Linear(64, 128, bias=True, weight_initializer='glorot')
        # PyG2.0.4 upstream SAGPooling has no extra select scalar.
        self.readout.select.weight.data.fill_(1.)
        self.readout.select.weight.requires_grad_(False)

    def forward(self, h, t):
        # Original ne_update order.
        h.x = self.feature_conv(h.x, h.edge_index, h.edge_attr)
        t.x = self.feature_conv(t.x, t.edge_index, t.edge_attr)
        h.x = F.elu(self.norm(h.x, h.batch))
        t.x = F.elu(self.norm(t.x, t.batch))
        h.edge_attr = self.lin_up(h.edge_attr)
        t.edge_attr = self.lin_up(t.edge_attr)
        h.edge_attr = F.elu(h.edge_attr)
        t.edge_attr = F.elu(t.edge_attr)
        # Original second convolution and bond update, then GlobalPool.
        h.x = self.feature_conv2(h.x, h.edge_index, h.edge_attr)
        t.x = self.feature_conv2(t.x, t.edge_index, t.edge_attr)
        h.edge_attr = self.lin_up2(h.edge_attr)
        t.edge_attr = self.lin_up2(t.edge_attr)
        h_atoms, _, _, h_batch, _, _ = self.readout(h.x, h.edge_index, edge_attr=h.edge_attr, batch=h.batch)
        t_atoms, _, _, t_batch, _, _ = self.readout(t.x, t.edge_index, edge_attr=t.edge_attr, batch=t.batch)
        h_pool = global_add_pool(h_atoms, h_batch, size=h.num_graphs)
        t_pool = global_add_pool(t_atoms, t_batch, size=t.num_graphs)
        # Original h_data_edge batch: one row per directed bond, molecule order.
        h_bond = global_add_pool(h.edge_attr, h.batch[h.edge_index[0]], size=h.num_graphs)
        t_bond = global_add_pool(t.edge_attr, t.batch[t.edge_index[0]], size=t.num_graphs)
        h_bond = F.elu(self.re_shape_e(h_bond))
        t_bond = F.elu(self.re_shape_e(t_bond))
        h_pool, t_pool = h_pool * h_bond, t_pool * t_bond
        h_pool = F.elu(F.normalize(h_pool, p=2, dim=1))
        t_pool = F.elu(F.normalize(t_pool, p=2, dim=1))
        h.x = F.elu(self.norm2(h.x, h.batch))
        t.x = F.elu(self.norm2(t.x, t.batch))
        h.edge_attr, t.edge_attr = F.elu(h.edge_attr), F.elu(t.edge_attr)
        return h, t, h_pool, t_pool


class CSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_node_feature = Linear(55, 128, bias=True, weight_initializer='glorot')
        self.initial_edge_feature = Linear(17, 64, bias=True, weight_initializer='glorot')
        self.initial_node_norm = LayerNorm(128)
        self.blocks = nn.ModuleList([CSEBlock(), CSEBlock()])

    def forward(self, batch):
        # clone keeps the immutable molecular cache unmodified.
        h, t = batch['graph_a'].clone(), batch['graph_b'].clone()
        h.x = self.initial_node_feature(h.x)
        t.x = self.initial_node_feature(t.x)
        h.x = self.initial_node_norm(h.x, h.batch)
        t.x = self.initial_node_norm(t.x, t.batch)
        h.x, t.x = F.elu(h.x), F.elu(t.x)
        h.edge_attr = self.initial_edge_feature(h.edge_attr)
        t.edge_attr = self.initial_edge_feature(t.edge_attr)
        h.edge_attr, t.edge_attr = F.elu(h.edge_attr), F.elu(t.edge_attr)
        repr_h, repr_t = [], []
        for block in self.blocks:
            h, t, rh, rt = block(h, t)
            repr_h.append(rh)
            repr_t.append(rt)
        return .6 * repr_h[0] + .4 * repr_h[-1], .6 * repr_t[0] + .4 * repr_t[-1]


class RareDDIEControl(nn.Module):
    def __init__(self, assets):
        super().__init__()
        self.fin_entities = nn.Embedding.from_pretrained(assets['fin_entities'].clone(), freeze=False)
        self.fin_relations = nn.Embedding.from_pretrained(assets['fin_relations'].clone(), freeze=False)
        self.register_buffer('neighbor_entities', assets['neighbor_entities'])
        self.register_buffer('neighbor_relations', assets['neighbor_relations'])
        self.cse = CSE()
        self.nai = NAI()
        self.pvr = PairVAE(256)
        self.classifier = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 256), nn.GELU(),
                                        nn.Dropout(.2), nn.Linear(256, 85))

    def neighbors(self, drugs):
        entity, relation = self.neighbor_entities[drugs], self.neighbor_relations[drugs]
        valid = (entity >= 0) & (relation >= 0)
        mask = valid[:, :, None]
        return self.fin_entities(entity.clamp_min(0)) * mask, self.fin_relations(relation.clamp_min(0)) * mask, valid

    def forward(self, batch):
        head, tail = self.cse(batch)
        # Original uses the SAME right-minus-left guide on BOTH endpoints.
        guide = tail - head
        nai_h = self.nai(*self.neighbors(batch['drug_a']), head, guide)
        nai_t = self.nai(*self.neighbors(batch['drug_b']), tail, guide)
        pair = torch.cat([nai_h, nai_t], -1)
        mu, rec, kl = self.pvr(pair, pair.new_empty((len(pair), 0), dtype=torch.bool))
        # Deterministic mu_ab classifier, as in the no-path comparator.
        return {'logits': self.classifier(mu), 'rec': rec, 'kl': kl}


GROUPS = {'fin_entities': ('fin_entities.',), 'fin_relations': ('fin_relations.',), 'cse': ('cse.',),
          'nai': ('nai.',), 'pvr': ('pvr.',), 'classifier': ('classifier.',)}
FORBIDDEN = ('inter', 'path', 'gat', 'cross')


def parameter_inventory(model):
    """Every named parameter with its group; proves no DSN/path parameters exist."""
    rows = [{'name': name, 'shape': list(p.shape), 'numel': p.numel(), 'requires_grad': p.requires_grad,
             'group': next(g for g, prefixes in GROUPS.items() if name.startswith(prefixes))}
            for name, p in model.named_parameters()]
    lowered = [r['name'].lower() for r in rows]
    convs = {name: type(m).__name__ for name, m in model.cse.named_modules() if isinstance(m, TransformerConv)}
    return {
        'parameters': sum(r['numel'] for r in rows),
        'trainable_parameters': sum(r['numel'] for r in rows if r['requires_grad']),
        'groups': {g: sum(r['numel'] for r in rows if r['group'] == g) for g in GROUPS},
        'frozen': [r['name'] for r in rows if not r['requires_grad']],
        'forbidden_name_matches': [n for n in lowered if any(word in n for word in FORBIDDEN)],
        'dsn_inter_attention_present': any('interatt' in n for n in lowered),
        'inter_projection_present': any('inter_projection' in n for n in lowered),
        'path_modules_present': any(n.startswith('path.') for n in lowered),
        'molecular_transformer_convs': convs,
        'fin_trainable': model.fin_entities.weight.requires_grad and model.fin_relations.weight.requires_grad,
        'pair_input': model.pvr.feature_dim, 'latent': model.pvr.mu.out_features,
        'classifier': [str(m) for m in model.classifier],
        'tensors': rows}

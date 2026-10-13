"""Original bond-aware RareDDIE order with the exact DSN inter-graph layer.

All pair-conditioned states are freshly computed with gradients. The optional
path branch ports EmerGNN's multiplicative sum propagation and shared ordered
pair gates; only real cached topology, never learned states, is reused.

GPU3 speed copy: the inter-graph GAT runs once per direction on the packed
disconnected bipartite batch. The collator already applied each pair's local
default-GATConv self-loop rule, so automatic loops are disabled only here.
"""
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, SAGPooling, LayerNorm, Linear, global_add_pool, GATConv


class InterGraphAttention(nn.Module):
    # Original class, references/DSN-DDI/drugbank_test/layers.py. Same
    # parameters; per-pair self loops arrive explicitly in the packed edges.
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv((input_dim, input_dim), 32, 2, add_self_loops=False)

    def forward(self, h_data, t_data, b_graph):
        edge_index = b_graph.edge_index
        h_input = F.elu(h_data.x)
        t_input = F.elu(t_data.x)
        t_rep = self.inter((h_input, t_input), edge_index)
        h_rep = self.inter((t_input, h_input), edge_index[[1, 0]])
        return h_rep, t_rep


class CSEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # RareDDIE MVN_DDI_Block: two 2x64 TransformerConvs, bond64.
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
        self.interAtt = InterGraphAttention(128)
        self.inter_projection = Linear(192, 128, bias=True, weight_initializer='glorot')

    def inter_pair_batch(self, h, t, cross_edges):
        # One call on the packed batch: cross_edges holds only within-pair
        # edges (global offsets) with each pair's own min(n_h, n_t) local loops.
        return self.interAtt(SimpleNamespace(x=h.x), SimpleNamespace(x=t.x),
                             SimpleNamespace(edge_index=cross_edges))

    def forward(self, h, t, cross_edges):
        # Original ne_update order.
        h.x = self.feature_conv(h.x, h.edge_index, h.edge_attr)
        t.x = self.feature_conv(t.x, t.edge_index, t.edge_attr)
        h.x = F.elu(self.norm(h.x, h.batch))
        t.x = F.elu(self.norm(t.x, t.batch))
        h.edge_attr = self.lin_up(h.edge_attr)
        t.edge_attr = self.lin_up(t.edge_attr)
        h.edge_attr = F.elu(h.edge_attr)
        t.edge_attr = F.elu(t.edge_attr)
        # Original second convolution and bond update before GlobalPool.
        h.x = self.feature_conv2(h.x, h.edge_index, h.edge_attr)
        t.x = self.feature_conv2(t.x, t.edge_index, t.edge_attr)
        h.edge_attr = self.lin_up2(h.edge_attr)
        t.edge_attr = self.lin_up2(t.edge_attr)
        hi, ti = self.inter_pair_batch(h, t, cross_edges)
        h.x = self.inter_projection(torch.cat([h.x, hi], dim=-1))
        t.x = self.inter_projection(torch.cat([t.x, ti], dim=-1))
        h_atoms, _, _, h_batch, _, _ = self.readout(h.x, h.edge_index, edge_attr=h.edge_attr, batch=h.batch)
        t_atoms, _, _, t_batch, _, _ = self.readout(t.x, t.edge_index, edge_attr=t.edge_attr, batch=t.batch)
        h_pool = global_add_pool(h_atoms, h_batch, size=h.num_graphs)
        t_pool = global_add_pool(t_atoms, t_batch, size=t.num_graphs)
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
        # clone keeps the immutable molecular cache unmodified; no deepcopy of
        # learned features occurs anywhere after this point.
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
            h, t, rh, rt = block(h, t, batch['cross_edges'])
            repr_h.append(rh)
            repr_t.append(rt)
        return .6 * repr_h[0] + .4 * repr_h[-1], .6 * repr_t[0] + .4 * repr_t[-1]


class NAI(nn.Module):
    """RareDDIE neighbor_encoder with an explicit padding softmax mask."""
    def __init__(self):
        super().__init__()
        self.gcn_w = nn.Linear(256, 128)
        nn.init.xavier_normal_(self.gcn_w.weight)
        self.dropout = nn.Dropout(.5)
        self.Bilinear = nn.Bilinear(128, 128, 1, bias=False)
        self.Linear_self = nn.Linear(128, 128, bias=False)
        self.Linear_nei = nn.Linear(128, 128, bias=False)
        self.Linear_weak_rel = nn.Linear(128, 128, bias=False)

    def forward(self, entity, relation, valid, feature, weak_rel):
        out = self.gcn_w(torch.cat([self.dropout(relation), self.dropout(entity)], -1))
        guide = self.Linear_weak_rel(weak_rel[:, None].expand_as(out))
        score = self.Bilinear(guide, out).squeeze(-1)
        # All-empty rows produce zero neighbor context, no biased padding.
        score = score.masked_fill(~valid, -torch.inf)
        score = torch.where(valid.any(1, keepdim=True), score, torch.zeros_like(score))
        attention = score.softmax(1) * valid.to(score.dtype)
        context = torch.bmm(attention[:, None], out).squeeze(1)
        return F.elu(self.Linear_nei(context) + self.Linear_self(feature))


class EmerGNNPath(nn.Module):
    def __init__(self, n_rel, depth=4):
        super().__init__()
        self.depth = depth
        self.rel_kg = nn.ModuleList([nn.Embedding(2 * n_rel + 1, 128) for _ in range(depth)])
        self.linear = nn.ModuleList([nn.Linear(128, 128) for _ in range(depth)])
        # Original init_weight affects relation embeddings and layer projections
        # before relation gate modules are constructed, leaving gate init default.
        for module in [self.rel_kg, self.linear]:
            for parameter in module.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.relation_linear = nn.ModuleList([nn.Linear(256, 5) for _ in range(depth)])
        self.attn_relation = nn.ModuleList([nn.Linear(5, 2 * n_rel + 1) for _ in range(depth)])

    def propagate(self, source, source_nodes, target_nodes, graph, gates):
        n_nodes = int(graph['n_nodes'])
        state = source.new_zeros(n_nodes, 128).index_copy(0, source_nodes, source)
        reachable = torch.zeros(n_nodes, dtype=torch.bool, device=source.device)
        reachable[source_nodes] = True
        src, dst, rel = graph['edge_index'][0], graph['edge_index'][1], graph['relations']
        edge_pair, multiplicity = graph['edge_pair'], graph['multiplicity']
        for layer in range(self.depth):
            # Exactly gated elementwise multiplication, SUM, linear, ReLU.
            relation_input = gates[layer][edge_pair, rel, None] * self.rel_kg[layer](rel)
            message = state[src] * relation_input * multiplicity[:, None]
            aggregated = state.new_zeros(n_nodes, 128).index_add(0, dst, message)
            reach_count = torch.zeros(n_nodes, dtype=torch.long, device=source.device)
            reach_count.index_add_(0, dst, reachable[src].long())
            reachable = reach_count > 0
            # Original linear bias can create features at unreachable nodes.
            # Apply it only after a real source-reachable edge delivered a state.
            state = F.relu(self.linear[layer](aggregated)) * reachable[:, None]
        present = reachable[target_nodes] & graph['path_mask']
        return state[target_nodes] * present[:, None], present

    def forward(self, head, tail, graph):
        ordered_pair = torch.cat([head, tail], -1)
        # Both directions use these SAME gates from the SAME ordered pair.
        gates = [torch.sigmoid(attn(F.relu(proj(ordered_pair))))
                 for proj, attn in zip(self.relation_linear, self.attn_relation)]
        tail_hid, ht = self.propagate(head, graph['head_nodes'], graph['tail_nodes'], graph, gates)
        head_hid, th = self.propagate(tail, graph['tail_nodes'], graph['head_nodes'], graph, gates)
        # Original readout order: [head_hid (v->u), tail_hid (u->v)].
        return torch.cat([head_hid, tail_hid], -1), torch.stack([th, ht], -1)


class PairVAE(nn.Module):
    def __init__(self, feature_dim, presence_dim=0):
        super().__init__()
        self.feature_dim = feature_dim
        self.presence_dim = presence_dim
        self.encoder = nn.Sequential(nn.Linear(feature_dim + presence_dim, 256), nn.GELU())
        self.mu = nn.Linear(256, 128)
        self.logvar = nn.Linear(256, 128)
        self.decoder = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, feature_dim))

    def forward(self, pair, path_masks):
        # Equalize each continuous 128-dimensional branch at the PVR boundary;
        # CSE/NAI/path formulas upstream remain unchanged. No learned affine.
        branches = pair.reshape(len(pair), self.feature_dim // 128, 128)
        valid = torch.ones(branches.shape[:2], dtype=torch.bool, device=pair.device)
        if self.presence_dim:
            valid = torch.cat([valid[:, :2], path_masks], dim=1)
        branches = F.layer_norm(branches, (128,)) * valid[:, :, None]
        continuous = branches.flatten(1)
        encoder_input = (torch.cat([continuous, path_masks.to(pair.dtype)], dim=1)
                         if self.presence_dim else continuous)
        encoded = self.encoder(encoder_input)
        mu, logvar = self.mu(encoded), self.logvar(encoded)
        z = mu + torch.exp(.5 * logvar) * torch.randn_like(mu)
        reconstructed = self.decoder(z)
        branch_mse = (reconstructed.reshape_as(branches) - branches.detach()).square().mean(-1)
        # Average over the actual valid branches PER SAMPLE, then over samples.
        # Every pair has equal weight regardless of missing path directions.
        # Flags and absent continuous branches have no reconstruction loss.
        rec = ((branch_mse * valid).sum(1) / valid.sum(1)).mean()
        kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
        return mu, rec, kl


class DSNRareDDI(nn.Module):
    def __init__(self, assets, config):
        super().__init__()
        self.fin_entities = nn.Embedding.from_pretrained(assets['fin_entities'].clone(), freeze=False)
        self.fin_relations = nn.Embedding.from_pretrained(assets['fin_relations'].clone(), freeze=False)
        self.register_buffer('neighbor_entities', assets['neighbor_entities'])
        self.register_buffer('neighbor_relations', assets['neighbor_relations'])
        self.cse = CSE()
        self.nai = NAI()
        self.with_path = config['with_path']
        if self.with_path:
            self.path = EmerGNNPath(len(assets['fin_relation_ids']), config['path_depth'])
        self.pvr = PairVAE(512, 2) if self.with_path else PairVAE(256)
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
        masks = torch.empty((len(head), 0), dtype=torch.bool, device=head.device)
        if self.with_path:
            # Original source is a projected molecular vector; adapt to the live
            # DSN-conditioned CSE molecular vector, keeping source-only init.
            path, masks = self.path(head, tail, batch['path_graph'])
            pair = torch.cat([pair, path], -1)
        mu, rec, kl = self.pvr(pair, masks)
        return {'logits': self.classifier(mu), 'rec': rec, 'kl': kl, 'path_mask': masks}

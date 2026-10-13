"""Label-independent bounded real routes from the complete non-DDI DRKG.

Topology only is cached. A two-hop ball intersection is an exact test for an
undirected path of length <=4 (inverse relations are deliberately present).
We select meeting nodes by entity type, then up to two real two-hop arms per
endpoint. No local NAI slots, event labels, or learned features enter extraction.
"""
from functools import lru_cache
import numpy as np
from scipy import sparse


def spread(values, limit):
    if len(values) <= limit:
        return values
    return values[np.linspace(0, len(values) - 1, limit, dtype=np.int64)]


def balanced(values, types, limit):
    """Round-robin entity types; spaced source indices within each type."""
    groups = [values[types[values] == t] for t in np.unique(types[values])]
    groups = [spread(g, limit) for g in groups]
    selected = []
    for slot in range(limit):
        for group in groups:
            if slot < len(group):
                selected.append(group[slot])
                if len(selected) == limit:
                    return np.asarray(selected, dtype=np.int32)
    return np.asarray(selected, dtype=np.int32)


class RouteExtractor:
    def __init__(self, triples, entities, n_rel, meeting_cap=32, arm_cap=2):
        self.n_rel, self.meeting_cap, self.arm_cap = n_rel, meeting_cap, arm_cap
        # triples are (head, relation, tail). Preserve source row multiplicity.
        h, r, t = triples.T
        self.src = np.concatenate([h, t]).astype(np.int32)
        self.dst = np.concatenate([t, h]).astype(np.int32)
        self.rel = np.concatenate([r, r + n_rel]).astype(np.int16)
        order = np.lexsort((self.rel, self.dst, self.src))
        self.src, self.dst, self.rel = self.src[order], self.dst[order], self.rel[order]
        self.ptr = np.concatenate([[0], np.bincount(self.src, minlength=len(entities)).cumsum()])
        # sum_duplicates compacts CSR indices in place. Keep the typed source
        # arrays independent, since their row offsets preserve parallel facts.
        self.adj = sparse.csr_matrix((np.ones(len(self.src), np.int32),
                                     self.dst, self.ptr), shape=(len(entities), len(entities)), copy=True)
        self.adj.sum_duplicates()
        self.adj.data[:] = 1
        names = [e.split('::', 1)[0] for e in entities]
        type_names = sorted(set(names))
        self.type_names = type_names
        self.types = np.asarray([type_names.index(t) for t in names], np.int8)
        self.balls = {}

    def neighbors(self, node):
        return self.adj.indices[self.adj.indptr[node]:self.adj.indptr[node + 1]]

    def build_balls(self, endpoints):
        for node in endpoints:
            row = self.adj.getrow(int(node))
            two = (row @ self.adj).indices
            self.balls[int(node)] = np.unique(np.concatenate([[node], row.indices, two])).astype(np.int32)

    def edges_between(self, u, v):
        start, end = self.ptr[u:u + 2]
        dest = self.dst[start:end]
        left, right = np.searchsorted(dest, v, side='left'), np.searchsorted(dest, v, side='right')
        return range(int(start + left), int(start + right))

    @lru_cache(maxsize=65536)
    def arm(self, endpoint, mid):
        if endpoint == mid:
            return ()
        direct = tuple(self.edges_between(endpoint, mid))
        if direct:
            return direct
        bridges = np.intersect1d(self.neighbors(endpoint), self.neighbors(mid), assume_unique=True)
        bridges = balanced(bridges, self.types, self.arm_cap)
        edge_ids = []
        for bridge in bridges:
            edge_ids.extend(self.edges_between(endpoint, int(bridge)))
            edge_ids.extend(self.edges_between(int(bridge), mid))
        return tuple(edge_ids)

    def extract(self, head, tail):
        if head < 0 or tail < 0:
            return np.asarray([head, tail], np.int32), np.empty((0, 4), np.int32), False
        mids = np.intersect1d(self.balls[head], self.balls[tail], assume_unique=True)
        if not len(mids):
            return np.asarray([head, tail], np.int32), np.empty((0, 4), np.int32), False
        mids = balanced(mids, self.types, self.meeting_cap)
        edge_ids = set()
        for mid in mids:
            edge_ids.update(self.arm(head, int(mid)))
            edge_ids.update(self.arm(tail, int(mid)))
        # Include inverse of every selected real row. Because rows are sorted,
        # retrieve its reverse by endpoints and relation (including duplicates).
        selected = sorted(edge_ids)
        triples = {(int(self.src[i]), int(self.dst[i]), int(self.rel[i])) for i in selected}
        inverse = {(v, u, r + self.n_rel if r < self.n_rel else r - self.n_rel) for u, v, r in triples}
        topology = sorted(triples | inverse)
        # Multiplicity of repeated source facts follows the original sparse KG.
        counts = {}
        for u, v, r in topology:
            ids = self.edges_between(u, v)
            counts[u, v, r] = sum(int(self.rel[i]) == r for i in ids)
        nodes = np.asarray(sorted({head, tail} | {u for u, v, r in topology} | {v for u, v, r in topology}), np.int32)
        local = {int(n): i for i, n in enumerate(nodes)}
        edges = [(local[u], local[v], r, counts[u, v, r]) for u, v, r in topology]
        edges += [(i, i, 2 * self.n_rel, 1) for i in range(len(nodes))]
        return nodes, np.asarray(edges, np.int32).reshape(-1, 4), True

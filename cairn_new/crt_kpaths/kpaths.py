"""K-Paths diverse path retrieval (Algorithm 1) over the complete non-DDI DRKG.

Official source: references/K-Paths, k-paths/utils/Kpaths_utils.py
(rsinghlab/K-Paths bf25d24eac6939574542d8994957b340ea1f5ab8): find_Kpaths,
one_node_in_graph, two_nodes_in_graph. Parameter adaptation: K=10, L=4 (official
default L=3) to keep the existing 4-hop EmerGNN path branch.

Background: every non-DDI fact (h, r, t) is the directed edge h->t with r and an
explicit inverse t->h with r+105. Official preprocessing
(augmented_network_utils.create_augmentedKG -> merge_duplicate_relations) first
merges the parallel relations of each node pair into one composite relation ID
("5_10"), so its DiGraph edges already carry composite semantics, and
get-subgraph.py (subgraph_utils.expand_relations) splits selected composites
back into atomic relations. Adapted here to the existing DRKG relation IDs: each
directed node pair carries the sorted tuple of its distinct atomic relations,
formed after adding inverses ((v,u) carries the inverse tuple); selected hops
are expanded back to atomic DRKG facts with multiplicity; no composite
embeddings are learned. Nodes/edges are inserted in global node-ID order.

Search: the official calls run on this complete DiGraph (6.0M directed node
pairs) built ONCE per process before forking workers; equal-length order is
therefore networkx's own discovery order on the full graph, exactly as the
official code. The only adaptation is an explicit budget of 1000 examined
<= L node paths; a budget stop is labeled (is_exact=False), never "no path".
"""
import time
from collections import Counter
import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

FAR = 127  # distance > depth of the stored endpoint balls


class Background:
    def __init__(self, triples, entities, n_rel):
        n = len(entities)
        self.n_entities, self.n_rel, self.self_rel = n, n_rel, 2 * n_rel
        h, r, t = (triples[:, i].astype(np.int64) for i in range(3))
        src, dst, rel = np.concatenate([h, t]), np.concatenate([t, h]), np.concatenate([r, r + n_rel])
        order = np.lexsort((rel, dst, src))
        src, dst, rel = src[order], dst[order], rel[order]
        # Atomic directed relations with source-fact multiplicity.
        new = np.r_[True, (src[1:] != src[:-1]) | (dst[1:] != dst[:-1]) | (rel[1:] != rel[:-1])]
        starts = np.flatnonzero(new)
        self.atom_src, self.atom_dst, self.atom_rel = src[starts], dst[starts], rel[starts].astype(np.int32)
        self.atom_mult = np.diff(np.r_[starts, len(src)]).astype(np.int32)
        # Directed node pairs; atoms of a pair are contiguous and sorted by relation.
        new = np.r_[True, (self.atom_src[1:] != self.atom_src[:-1]) | (self.atom_dst[1:] != self.atom_dst[:-1])]
        starts = np.flatnonzero(new)
        self.pair_src, self.pair_dst = self.atom_src[starts], self.atom_dst[starts]
        self.pair_atom_ptr = np.r_[starts, len(self.atom_src)]
        self.ptr = np.r_[0, np.bincount(self.pair_src, minlength=n).cumsum()]
        # Signature IDs: a single relation r is ID r; composite tuples are numbered
        # in global (src, dst) order. Only equality of signatures is ever used.
        sizes = np.diff(self.pair_atom_ptr)
        self.pair_sig = np.where(sizes == 1, self.atom_rel[starts], -1).astype(np.int64)
        self.signatures = {r: (r,) for r in range(2 * n_rel)}
        index = {}
        for p in np.flatnonzero(sizes > 1):
            key = tuple(self.atom_rel[self.pair_atom_ptr[p]:self.pair_atom_ptr[p + 1]].tolist())
            if key not in index:
                index[key] = 2 * n_rel + 1 + len(index)
                self.signatures[index[key]] = key
            self.pair_sig[p] = index[key]
        self.in_graph = np.bincount(np.concatenate([h, t]), minlength=n) > 0
        self.adj = sparse.csr_matrix((np.ones(len(self.pair_dst), np.float32), self.pair_dst, self.ptr), shape=(n, n))
        # The inverse-augmented DiGraph is symmetric: has_path == same component.
        _, self.component = connected_components(self.adj, directed=False)
        types = np.asarray([e.split('::', 1)[0] for e in entities])
        self.type_names = sorted(set(types))
        self.types = np.searchsorted(self.type_names, types).astype(np.int8)

    def distances(self, source, depth=3):
        dist = np.full(self.n_entities, FAR, np.int8)
        if source < 0 or not self.in_graph[source]:
            return dist
        dist[source] = 0
        frontier = np.zeros(self.n_entities, np.float32)
        frontier[source] = 1
        for level in range(1, depth + 1):
            reached = (self.adj @ frontier > 0) & (dist == FAR)
            dist[reached] = level
            frontier = reached.astype(np.float32)
        return dist

    def exact_distance(self, a, b, da, db):
        """True shortest hop distance (None if disconnected or absent)."""
        if a < 0 or b < 0 or not (self.in_graph[a] and self.in_graph[b]) or self.component[a] != self.component[b]:
            return None
        best = int((da.astype(np.int16) + db).min())
        if best <= 6:  # some midpoint of a <= 6 path is within 3 of both ends
            return best
        seen = np.zeros(self.n_entities, bool)
        seen[a] = True
        frontier, level = seen.astype(np.float32), 0
        while not seen[b]:
            reached = (self.adj @ frontier > 0) & ~seen
            seen |= reached
            frontier, level = reached.astype(np.float32), level + 1
        return level

    def pair_index(self, u, v):
        start, end = self.ptr[u], self.ptr[u + 1]
        return start + int(np.searchsorted(self.pair_dst[start:end], v))

    def full_graph(self):
        """Complete inverse-augmented DiGraph; edge 'relation' = signature ID."""
        graph = nx.DiGraph()
        graph.add_nodes_from(np.flatnonzero(self.in_graph).tolist())
        graph.add_edges_from((u, v, {'relation': s}) for u, v, s in zip(
            self.pair_src.tolist(), self.pair_dst.tolist(), self.pair_sig.tolist()))
        return graph

    def neighbor_paths(self, node, max_paths):
        """official one_node_in_graph: out-neighbors, first of each signature."""
        seen, paths = set(), []
        for p in range(self.ptr[node], self.ptr[node + 1]):
            sig = int(self.pair_sig[p])
            if sig not in seen:
                seen.add(sig)
                paths.append([int(node), int(self.pair_dst[p])])
            if len(paths) >= max_paths:
                break
        return paths

    def path_signatures(self, path):
        return [self.signatures[int(self.pair_sig[self.pair_index(u, v)])] for u, v in zip(path[:-1], path[1:])]


def diverse_paths(graph, a, b, max_length, max_paths, budget):
    """official two_nodes_in_graph loop plus an explicit candidate budget.

    Returns (selected node paths, candidates examined, exact). exact=False only
    when the budget stopped enumeration while another <= L path still existed.
    """
    seen, selected, candidates = set(), [], 0
    generator = nx.shortest_simple_paths(graph, a, b)
    for path in generator:
        if len(path) - 1 > max_length:
            return selected, candidates, True
        candidates += 1
        relations = tuple(graph[u][v]['relation'] for u, v in zip(path[:-1], path[1:]))
        if relations not in seen:
            seen.add(relations)
            selected.append(path)
        if len(selected) >= max_paths:
            return selected, candidates, True
        if candidates >= budget:
            # Peek only to label exactness; the peeked path is never used.
            following = next(generator, None)
            return selected, candidates, following is None or len(following) - 1 > max_length
    return selected, candidates, True


class Retriever:
    def __init__(self, background, graph, endpoint_distances, k=10, max_length=4, budget=1000,
                 max_nodes=130, max_edges=695):
        self.bg, self.graph, self.dist = background, graph, endpoint_distances
        self.k, self.max_length, self.budget = k, max_length, budget
        self.max_nodes, self.max_edges = max_nodes, max_edges

    def retrieve(self, a, b):
        bg, k, length = self.bg, self.k, self.max_length
        started = time.perf_counter()
        in_a = a >= 0 and bool(bg.in_graph[a])
        in_b = b >= 0 and bool(bg.in_graph[b])
        da = self.dist[a] if in_a else np.full(bg.n_entities, FAR, np.int8)
        db = self.dist[b] if in_b else np.full(bg.n_entities, FAR, np.int8)
        distance = bg.exact_distance(a, b, da, db) if in_a and in_b else None
        meta = {'fin_a': int(a), 'fin_b': int(b), 'a_in_graph': in_a, 'b_in_graph': in_b,
                'shortest_distance': -1 if distance is None else distance,
                'candidate_count': 0, 'is_exact': True, 'budget_truncated': False}
        if not in_a and not in_b:
            status, paths = 'both_absent', []
        elif in_a != in_b:
            status = 'one_end_absent'
            paths = bg.neighbor_paths(a if in_a else b, k)
        elif bg.component[a] != bg.component[b]:
            # official: nx.has_path False -> both neighbor lists (up to 2K)
            status, paths = 'disconnected_neighbors', bg.neighbor_paths(a, k) + bg.neighbor_paths(b, k)
        elif distance > length:
            # official: connected but shortest > L -> empty, no neighbor fallback
            status, paths = 'overlength_empty', []
        else:
            paths, meta['candidate_count'], exact = diverse_paths(self.graph, a, b, length, k, self.budget)
            meta['is_exact'], meta['budget_truncated'] = exact, not exact
            status = 'connected_paths'
        meta['status'] = status
        meta['fallback'] = status in ('one_end_absent', 'disconnected_neighbors')
        nodes, edges, ends, kept, reason = self.assemble(a, b, paths)
        meta.update(retrieved_paths=len(paths), selected_paths=kept, graph_budget_reason=reason,
                    graph_nodes=len(nodes), graph_edges=len(edges),
                    length_counts={n: c for n, c in sorted(Counter(len(p) - 1 for p in paths[:kept]).items())},
                    paths=[{'nodes': p, 'signatures': [list(s) for s in bg.path_signatures(p)]} for p in paths[:kept]])
        meta['path_mask'] = status == 'connected_paths' and kept > 0
        common = np.flatnonzero((da == 1) & (db == 1))
        meta['common_neighbors'] = {bg.type_names[t]: int((bg.types[common] == t).sum())
                                    for t in np.unique(bg.types[common])}
        meta['seconds'] = time.perf_counter() - started
        return nodes, edges, ends, meta

    def assemble(self, a, b, paths):
        """Union of selected hops' atomic edges plus inverses once, self loops once.

        Whole paths only, in retrieval order; a path that would exceed the
        original cache maxima (130 nodes / 695 edges) stops inclusion.
        """
        bg = self.bg
        node_set, atoms, kept, reason = {int(a), int(b)}, set(), 0, None
        for path in paths:
            new_nodes = node_set | set(path)
            new_atoms = set(atoms)
            for u, v in zip(path[:-1], path[1:]):
                for p in (bg.pair_index(u, v), bg.pair_index(v, u)):
                    new_atoms.update(range(bg.pair_atom_ptr[p], bg.pair_atom_ptr[p + 1]))
            if len(new_nodes) > self.max_nodes or len(new_atoms) + len(new_nodes) > self.max_edges:
                reason = f'path {kept + 1} would give {len(new_nodes)} nodes, {len(new_atoms) + len(new_nodes)} edges'
                break
            node_set, atoms, kept = new_nodes, new_atoms, kept + 1
        nodes = np.asarray(sorted(node_set), np.int32)
        local = {int(n): i for i, n in enumerate(nodes)}
        atoms = np.asarray(sorted(atoms), np.int64)
        edges = [(local[int(bg.atom_src[i])], local[int(bg.atom_dst[i])], int(bg.atom_rel[i]), int(bg.atom_mult[i]))
                 for i in atoms]
        edges.sort()
        edges += [(i, i, bg.self_rel, 1) for i in range(len(nodes))]
        return nodes, np.asarray(edges, np.int32).reshape(-1, 4), [local[int(a)], local[int(b)]], kept, reason

"""Memory Graph — GNN-inspired associative memory with spreading activation.

Neural network parallel:
    - Memory entries = nodes (neurons)
    - Association weights = edges (synaptic connections)
    - Spreading activation = forward propagation (message passing)
    - Multi-hop recall = multi-layer GNN

Equation 4 — Association (Edge Weight):
    w_ij = (α_k · sim(k_i, k_j) + α_v · sim(v_i, v_j)) · exp(-|t_i - t_j| / τ)

    Combines semantic similarity (shared topics/entities) with temporal
    proximity (mentioned in same conversation). Like how synaptic
    connections form between co-activated neurons (Hebbian learning).

Equation 5 — Spreading Activation (GNN Message Passing):
    a_i^(0) = s_i · sim(q, k_i)                          [initial query activation]
    a_i^(l+1) = a_i^(l) + η · Σ_j(w_ij · a_j^(l))      [propagation per hop]

    After L hops, return all memories where a_i^(L) > θ_activate.

    This is literally one step of GNN message passing — activation flows
    through the graph like signal through a neural network.

Equation 5b — Degree-Normalized Spreading (GCN, Kipf & Welling 2017):
    a_i^(l+1) = a_i^(l) + η · (1/|N(i)|) · Σ_j(w_ij · a_j^(l))

    Mean aggregation instead of sum — prevents high-degree nodes from
    dominating. A node with 5 neighbors gets the same boost magnitude
    as a node with 1 neighbor.

Example:
    "Harshal" ←→ "Mandar in bhusawal" ←→ "we all study at VIT"
                                        ←→ "Raj in dhule"

    Query: "who studies at VIT?"
    Hop 0: "VIT" node activates strongly
    Hop 1: Activation spreads to Mandar, Raj through edges
    Result: LLM sees all connected memories
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from itm.config import MemoryConfig
from itm.embeddings import EmbeddingService

if TYPE_CHECKING:
    from itm.core import MemoryEntry


class MemoryGraph:
    """Graph of memory associations enabling multi-hop recall.

    Stores edges as a sparse adjacency dict: {i: {j: weight}}.
    Edges are undirected (w_ij = w_ji).
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        # Sparse adjacency: {node_i: {node_j: weight}}
        self.edges: dict[int, dict[int, float]] = {}

    def add_memory(self, index: int, memories: list[MemoryEntry]):
        """Compute edges from a new memory to all existing memories.

        Called whenever a new memory is created. Computes Equation 4
        (association weight) against every existing memory and stores
        edges that exceed θ_edge.
        """
        new_mem = memories[index]
        self.edges.setdefault(index, {})

        for j, other in enumerate(memories):
            if j == index:
                continue

            weight = self._compute_edge_weight(new_mem, other)
            if weight >= self.config.theta_edge:
                self.edges[index][j] = weight
                self.edges.setdefault(j, {})
                self.edges[j][index] = weight  # undirected

    @staticmethod
    def _displacement(mem: MemoryEntry) -> np.ndarray:
        """Equation 18: Compute displacement vector d_i = normalize(v_i - k_i).

        NN parallel: ResNet skip connections — the residual carries
        the relationship type. "raj lives in dhule" → displacement
        captures "person→location" pattern regardless of actual names.
        """
        d = mem.value - mem.key
        norm = np.linalg.norm(d)
        if norm > 1e-8:
            return d / norm
        return d

    def _compute_edge_weight(self, m1: MemoryEntry, m2: MemoryEntry) -> float:
        """Equations 4 + 18: Association weight between two memories.

        Base (Eq 4):
        w_ij = (α_k · sim(k_i, k_j) + α_v · sim(v_i, v_j)) · exp(-|Δt|/τ)

        With displacement (Eq 18):
        w_ij = (α_k'·sim_k + α_v'·sim_v + α_d'·sim_d) · exp(-|Δt|/τ)

        Eq 18 adds displacement similarity: memories with the same
        relationship type (person→location, person→preference) get
        stronger edges even if the entities differ.

        The displacement weight is NOT added on top: that would push the max
        semantic weight to α_k + α_v + α_d ≈ 1.3 and admit ~30% more edges than
        theta_edge intends. Instead the three coefficients are renormalized to
        sum to 1.0 (convex combination), so the semantic term stays bounded in
        [-1, 1] (max 1.0) exactly as in the base case and theta_edge keeps its
        meaning. This matches the documented rescaling in config.py.
        """
        key_sim = EmbeddingService.cosine_similarity(m1.key, m2.key)
        val_sim = EmbeddingService.cosine_similarity(m1.value, m2.value)

        if self.config.displacement_edges_enabled:
            d1 = self._displacement(m1)
            d2 = self._displacement(m2)
            disp_sim = EmbeddingService.cosine_similarity(d1, d2)
            # Rescale α_k, α_v, α_d to a convex combination so the semantic
            # term stays bounded by 1.0 (theta_edge stays meaningful).
            total = self.config.alpha_k + self.config.alpha_v + self.config.alpha_d
            if total <= 1e-12:
                total = 1.0
            ak = self.config.alpha_k / total
            av = self.config.alpha_v / total
            ad = self.config.alpha_d / total
            semantic = ak * key_sim + av * val_sim + ad * disp_sim
        else:
            semantic = self.config.alpha_k * key_sim + self.config.alpha_v * val_sim

        time_diff = abs(m1.timestamp - m2.timestamp)
        temporal = np.exp(-time_diff / self.config.tau_temporal)

        return float(semantic * temporal)

    def spreading_activation(
        self,
        initial_activations: np.ndarray,
        n_hops: int | None = None,
        query_embedding: np.ndarray | None = None,
        keys: np.ndarray | None = None,
    ) -> np.ndarray:
        """Equations 5b + 20 + 21: Enhanced GNN message passing.

        Base (Eq 5b, GCN):
            a_i^(l+1) = a_i^(l) + η · (1/|N(i)|) · Σ_j(w_ij · a_j^(l))

        Eq 20 (JK-Net, Xu 2018): Multi-scale — combine activations from
        ALL depths, not just final. Local (1-hop) + global (4-hop).
            a_final = Σ_l (w_l · a^(l))
            η_l = η · damping^l  (prevents over-smoothing)

        Eq 21 (GAT, Veličković 2018): Query-aware messages — weight
        by sender's relevance to original query. Stops irrelevant
        but well-connected memories from stealing activation.
            message_ij = w_ij · a_j · sim(q, k_j)

        Args:
            initial_activations: (n,) array from multi-head scoring
            n_hops: propagation depth
            query_embedding: (d,) query vector for Eq 21 (optional)
            keys: (n, d) key matrix for Eq 21 (optional)

        Returns:
            (n,) array of final activations after propagation
        """
        multiscale = self.config.multiscale_enabled
        query_aware = (
            self.config.query_aware_spread_enabled
            and query_embedding is not None
            and keys is not None
        )

        if multiscale:
            n_hops = self.config.multiscale_hops
        elif n_hops is None:
            n_hops = self.config.n_hops

        # Eq 21: precompute query-key similarities for all memories
        if query_aware:
            q_key_sims = EmbeddingService.cosine_similarity_matrix(
                query_embedding, keys
            ).astype(np.float64)
            # Clamp to [0, 1] — only positive relevance contributes
            q_key_sims = np.maximum(q_key_sims, 0.0)
        else:
            q_key_sims = None

        activations = initial_activations.copy()
        n = len(activations)
        eta = self.config.eta_propagation

        # Eq 20: collect activations at each scale
        if multiscale:
            scale_activations = [activations.copy()]

        for hop in range(n_hops):
            # Eq 20: per-hop damping
            if multiscale:
                eta_l = eta * (self.config.multiscale_damping**hop)
            else:
                eta_l = eta

            # Message passing
            delta = np.zeros(n, dtype=np.float64)
            for i in range(n):
                if i not in self.edges:
                    continue
                neighbors = self.edges[i]
                if not neighbors:
                    continue
                neighbor_sum = 0.0
                neighbor_count = 0
                for j, weight in neighbors.items():
                    if j < n:
                        # Eq 21: query-aware message
                        if query_aware:
                            msg = weight * activations[j] * q_key_sims[j]
                        else:
                            msg = weight * activations[j]
                        neighbor_sum += msg
                        neighbor_count += 1
                if neighbor_count > 0:
                    delta[i] = neighbor_sum / neighbor_count

            activations = activations + eta_l * delta

            if multiscale:
                scale_activations.append(activations.copy())

        # Eq 20: JK-Net — combine all scales with uniform weights
        if multiscale and len(scale_activations) > 1:
            # Uniform weighting across all scales
            combined = np.zeros(n, dtype=np.float64)
            for scale_act in scale_activations:
                combined += scale_act
            activations = combined / len(scale_activations)

        return activations

    def recall_graph(
        self,
        query_embedding: np.ndarray,
        memories: list[MemoryEntry],
        strengths: np.ndarray,
    ) -> list[tuple[MemoryEntry, float]]:
        """Full graph-based recall: initial activation → spreading → threshold.

        Combines Equation 3 (initial relevance) with Equation 5 (spreading).

        Returns list of (MemoryEntry, activation) sorted descending.
        """
        if not memories:
            return []

        # Step 1: Initial activations — a_i^(0) = s_i · sim(q, k_i)
        keys = np.stack([m.key for m in memories])
        sims = EmbeddingService.cosine_similarity_matrix(query_embedding, keys)
        initial = strengths * sims

        # Step 2+3: Spread and collect
        return self.spread_and_collect(initial, memories)

    def spread_and_collect(
        self,
        initial_activations: np.ndarray,
        memories: list[MemoryEntry],
        query_embedding: np.ndarray | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Spreading activation + threshold collection.

        Accepts pre-computed initial activations (from multi-head scoring
        or classic s_i * sim), runs GNN propagation, and returns memories
        above activation threshold.

        This decouples activation computation from graph propagation,
        allowing Equations 9+11 (multi-head) to feed directly into
        Equations 5+20+21 (enhanced spreading activation).

        Args:
            initial_activations: (n,) from multi-head scoring
            memories: list of MemoryEntry
            query_embedding: (d,) for Eq 21 query-aware spreading (optional)
        """
        if not memories:
            return []

        # Eq 21: build keys matrix for query-aware spreading
        keys = None
        if self.config.query_aware_spread_enabled and query_embedding is not None:
            keys = np.stack([m.key for m in memories])

        # Spreading activation (Equations 5+20+21)
        final = self.spreading_activation(
            initial_activations,
            self.config.n_hops,
            query_embedding=query_embedding,
            keys=keys,
        )

        # Collect all memories above activation threshold
        indices = np.argsort(final)[::-1]
        results = []
        for idx in indices:
            idx = int(idx)
            act = float(final[idx])
            if act < self.config.activation_threshold:
                break
            memories[idx].access_count += 1
            results.append((memories[idx], act))

        return results

    def propagate_contradiction(
        self,
        new_index: int,
        memories: list[MemoryEntry],
        config: MemoryConfig,
    ) -> list[int]:
        """Equation 7: Graph-Propagated Contradiction.

        Like backpropagation through the memory graph — the "error signal"
        (value contradiction) flows through edges to update connected memories.

        For each graph neighbor j of new memory i:
            val_sim = sim(v_i, v_j)
            if val_sim < θ_value:
                v_j += b_j · (v_i - v_j)     [Equation 2 through graph]
                s_j *= δ_graph                [weaken contradicted neighbor]

        Returns list of corrected memory indices.
        """
        if new_index not in self.edges:
            return []

        new_mem = memories[new_index]
        corrected = []

        for j, edge_weight in self.edges[new_index].items():
            if j >= len(memories):
                continue

            neighbor = memories[j]
            val_sim = EmbeddingService.cosine_similarity(new_mem.value, neighbor.value)

            if val_sim < config.theta_value:
                # Contradiction detected through graph edge — apply Equation 2
                delta_v = new_mem.value - neighbor.value
                neighbor.value = neighbor.value + neighbor.bias * delta_v
                neighbor.strength *= config.delta_graph
                corrected.append(j)

        return corrected

    def rebuild(self, memories: list[MemoryEntry]):
        """Rebuild the entire graph from scratch.

        Useful after loading from disk or when the graph gets
        out of sync.
        """
        self.edges.clear()
        for i in range(len(memories)):
            self.edges.setdefault(i, {})
            for j in range(i):
                weight = self._compute_edge_weight(memories[i], memories[j])
                if weight >= self.config.theta_edge:
                    self.edges[i][j] = weight
                    self.edges.setdefault(j, {})
                    self.edges[j][i] = weight

    def adjust_edge_neighbors(self, index: int, factor: float):
        """Equation 16: Multiply all edges of a node by a factor.

        Used by the feedback loop to strengthen/weaken all connections
        of a memory based on whether the LLM used it in its response.

        factor > 1.0 → strengthen edges
        factor < 1.0 → weaken edges
        Edges below theta_edge are removed. Edges above 1.0 are capped.
        """
        if index not in self.edges:
            return
        to_remove = []
        for j in list(self.edges[index].keys()):
            new_weight = self.edges[index][j] * factor
            if new_weight < self.config.theta_edge:
                to_remove.append(j)
            else:
                new_weight = min(new_weight, 1.0)
                self.edges[index][j] = new_weight
                if j in self.edges and index in self.edges[j]:
                    self.edges[j][index] = new_weight  # undirected

        for j in to_remove:
            del self.edges[index][j]
            if j in self.edges and index in self.edges[j]:
                del self.edges[j][index]

    def get_neighbors(self, index: int) -> list[tuple[int, float]]:
        """Get all neighbors of a memory node, sorted by edge weight."""
        if index not in self.edges:
            return []
        neighbors = list(self.edges[index].items())
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors

    def node_count(self) -> int:
        return len(self.edges)

    def edge_count(self) -> int:
        total = sum(len(neighbors) for neighbors in self.edges.values())
        return total // 2  # undirected, each edge counted twice

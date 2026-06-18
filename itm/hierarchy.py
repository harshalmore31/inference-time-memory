"""Hierarchical Memory — L1→L2→L3 (MLP Hidden Layers + CNN Pooling).

Neural network parallel:
    Perceptron (1958) → flat, linear scan, O(n)
    MLP (Rumelhart 1986) → hidden layers, abstraction, O(log n)
    CNN (LeCun 1998) → pooling reduces dimensions while preserving features
    ResNet (He 2015) → skip connections prevent information loss in deep nets

Equation 22 — L2 Pattern Formation (CNN Weighted Pooling):
    Given cluster C = {m_i : avg pairwise sim > θ_cluster, |C| ≥ min_size}:
    K_L2 = Σ(s_i · k_i) / Σ(s_i)     [strength-weighted centroid key]
    V_L2 = Σ(s_i · v_i) / Σ(s_i)     [strength-weighted centroid value]
    S_L2 = Σ(s_i)                      [combined strength]

    This is weighted average pooling — same as CNN avg pooling but
    weighted by learned strengths. Stronger memories pull the centroid
    toward them, weaker ones contribute less.

Equation 23 — Hierarchical Recall with Skip Connections (ResNet):
    R(q) = w₁ · R_L1(q) + w₂ · R_L2(q) + w₃ · R_L3(q)

    Top-down drill: L3 → L2 → L1 for O(log n) at scale.
    Skip connections: all layers contribute for robustness.
    Without skip connections, if L3 misclassifies, L1 detail is lost.

Example:
    L1 memories:
      "Python list sorting"      s=1.82
      "Python dict comprehension" s=1.45
      "Python f-strings"         s=0.92
      "Python async await"       s=0.34

    → L2 pattern forms: "User works with Python"
      K_L2 = weighted centroid of 4 Python keys
      S_L2 = 4.53

    L2 patterns:
      "Works with Python"  S=4.53
      "Builds Django apps"  S=1.63
      "Studies AI/ML"       S=2.80

    → L3 identity forms: "Python developer studying AI"
      K_L3 = weighted centroid of L2 keys
      S_L3 = 8.96
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from itm.config import MemoryConfig
from itm.embeddings import EmbeddingService

if TYPE_CHECKING:
    from itm.core import MemoryEntry


@dataclass
class SemanticPattern:
    """L2 node — strength-weighted centroid of an L1 cluster.

    Equation 22: CNN weighted pooling applied to memory.
    Each L2 pattern summarizes a group of related L1 episodic memories.
    """

    key: np.ndarray  # (d,) centroid key embedding
    value: np.ndarray  # (d,) centroid value embedding
    strength: float  # sum of children strengths
    children: list[int]  # L1 memory indices


@dataclass
class IdentityNode:
    """L3 node — strength-weighted centroid of L2 patterns.

    Same pooling as L2 but one level up. Captures user identity
    and deep patterns (e.g., "Python web developer from Mumbai").
    """

    key: np.ndarray  # (d,) centroid key embedding
    value: np.ndarray  # (d,) centroid value embedding
    strength: float  # sum of L2 children strengths
    children: list[int]  # L2 pattern indices


class MemoryHierarchy:
    """Hierarchical memory: L1 (episodic) → L2 (semantic) → L3 (identity).

    Equations 22-23: CNN-like pooling for abstraction, ResNet skip
    connections for robustness, MLP-like top-down drill for speed.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.l2_patterns: list[SemanticPattern] = []
        self.l3_identities: list[IdentityNode] = []

    def consolidate(self, memories: list[MemoryEntry]):
        """Equation 22: Form L2 patterns from L1 clusters, L3 from L2.

        Agglomerative clustering: for each unassigned L1 memory, find
        neighbors with sim > θ_cluster. If cluster size ≥ min_cluster_size,
        form an L2 pattern (weighted centroid pooling).

        Called periodically (every consolidate_every updates).
        """
        if len(memories) < self.config.min_cluster_size:
            return

        # Build similarity matrix for all L1 keys
        keys = np.stack([m.key for m in memories])
        # Batch cosine similarity: keys are unit vectors, so dot = cosine
        sim_matrix = (keys @ keys.T).astype(np.float64)

        # Track which L1 memories are already in L2 patterns
        assigned: set[int] = set()
        for pattern in self.l2_patterns:
            for idx in pattern.children:
                if idx < len(memories):
                    assigned.add(idx)

        # Greedy clustering of unassigned L1 memories
        new_clusters: list[list[int]] = []
        visited: set[int] = set()

        for i in range(len(memories)):
            if i in assigned or i in visited:
                continue

            # Find neighbors with sim > θ_cluster
            cluster = [i]
            visited.add(i)
            for j in range(len(memories)):
                if j == i or j in assigned or j in visited:
                    continue
                if sim_matrix[i, j] > self.config.theta_cluster:
                    cluster.append(j)
                    visited.add(j)

            if len(cluster) < self.config.min_cluster_size:
                # Not enough to form a pattern — release them
                for idx in cluster:
                    visited.discard(idx)
                continue

            # Verify avg pairwise similarity within cluster
            total_sim = 0.0
            count = 0
            for a in range(len(cluster)):
                for b in range(a + 1, len(cluster)):
                    total_sim += sim_matrix[cluster[a], cluster[b]]
                    count += 1
            avg_sim = total_sim / count if count > 0 else 0.0

            if avg_sim >= self.config.theta_cluster:
                new_clusters.append(cluster)
                for idx in cluster:
                    assigned.add(idx)

        # Form L2 patterns from new clusters
        for cluster_indices in new_clusters:
            # Check for merge with existing L2
            merged = False
            for pattern in self.l2_patterns:
                overlap = set(cluster_indices) & set(pattern.children)
                if len(overlap) > len(cluster_indices) * 0.3:
                    # Merge into existing pattern
                    pattern.children = list(
                        set(pattern.children) | set(cluster_indices)
                    )
                    merged = True
                    break

            if not merged:
                pattern = self._form_l2(cluster_indices, memories)
                self.l2_patterns.append(pattern)

        # Update all L2 centroids
        for pattern in self.l2_patterns:
            self._update_l2_centroid(pattern, memories)

        # Form L3 identities from L2 patterns
        self._form_l3()

    def _form_l2(
        self, indices: list[int], memories: list[MemoryEntry]
    ) -> SemanticPattern:
        """Equation 22: Create L2 node via strength-weighted centroid pooling."""
        strengths = np.array([memories[i].strength for i in indices])
        keys = np.stack([memories[i].key for i in indices])
        values = np.stack([memories[i].value for i in indices])

        total_s = float(np.sum(strengths))
        if total_s < 1e-10:
            total_s = 1.0

        centroid_k = np.sum(strengths[:, None] * keys, axis=0) / total_s
        centroid_v = np.sum(strengths[:, None] * values, axis=0) / total_s

        # Normalize to unit sphere
        norm_k = np.linalg.norm(centroid_k)
        if norm_k > 1e-8:
            centroid_k /= norm_k
        norm_v = np.linalg.norm(centroid_v)
        if norm_v > 1e-8:
            centroid_v /= norm_v

        return SemanticPattern(
            key=centroid_k.astype(np.float32),
            value=centroid_v.astype(np.float32),
            strength=total_s,
            children=list(indices),
        )

    def _update_l2_centroid(
        self, pattern: SemanticPattern, memories: list[MemoryEntry]
    ):
        """Recompute L2 centroid from current children strengths."""
        valid = [i for i in pattern.children if i < len(memories)]
        if not valid:
            return

        strengths = np.array([memories[i].strength for i in valid])
        keys = np.stack([memories[i].key for i in valid])
        values = np.stack([memories[i].value for i in valid])

        total_s = float(np.sum(strengths))
        if total_s < 1e-10:
            total_s = 1.0

        pattern.key = (np.sum(strengths[:, None] * keys, axis=0) / total_s).astype(
            np.float32
        )
        pattern.value = (np.sum(strengths[:, None] * values, axis=0) / total_s).astype(
            np.float32
        )
        pattern.strength = total_s

        norm_k = np.linalg.norm(pattern.key)
        if norm_k > 1e-8:
            pattern.key /= norm_k
        norm_v = np.linalg.norm(pattern.value)
        if norm_v > 1e-8:
            pattern.value /= norm_v

    def _form_l3(self):
        """Form L3 identity nodes from L2 pattern clusters."""
        if len(self.l2_patterns) < self.config.min_identity_size:
            return

        l2_keys = np.stack([p.key for p in self.l2_patterns])
        sim_matrix = (l2_keys @ l2_keys.T).astype(np.float64)

        assigned: set[int] = set()
        for identity in self.l3_identities:
            for idx in identity.children:
                if idx < len(self.l2_patterns):
                    assigned.add(idx)

        new_clusters: list[list[int]] = []
        visited: set[int] = set()

        for i in range(len(self.l2_patterns)):
            if i in assigned or i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in range(len(self.l2_patterns)):
                if j == i or j in assigned or j in visited:
                    continue
                if sim_matrix[i, j] > self.config.theta_identity:
                    cluster.append(j)
                    visited.add(j)

            if len(cluster) >= self.config.min_identity_size:
                new_clusters.append(cluster)
                for idx in cluster:
                    assigned.add(idx)

        for cluster_indices in new_clusters:
            merged = False
            for identity in self.l3_identities:
                overlap = set(cluster_indices) & set(identity.children)
                if overlap:
                    identity.children = list(
                        set(identity.children) | set(cluster_indices)
                    )
                    merged = True
                    break
            if not merged:
                self.l3_identities.append(self._form_l3_node(cluster_indices))

        # Update all L3 centroids
        for identity in self.l3_identities:
            self._update_l3_centroid(identity)

    def _form_l3_node(self, l2_indices: list[int]) -> IdentityNode:
        """Create L3 node from L2 pattern cluster."""
        strengths = np.array([self.l2_patterns[i].strength for i in l2_indices])
        keys = np.stack([self.l2_patterns[i].key for i in l2_indices])
        values = np.stack([self.l2_patterns[i].value for i in l2_indices])

        total_s = float(np.sum(strengths))
        if total_s < 1e-10:
            total_s = 1.0

        centroid_k = np.sum(strengths[:, None] * keys, axis=0) / total_s
        centroid_v = np.sum(strengths[:, None] * values, axis=0) / total_s

        norm_k = np.linalg.norm(centroid_k)
        if norm_k > 1e-8:
            centroid_k /= norm_k
        norm_v = np.linalg.norm(centroid_v)
        if norm_v > 1e-8:
            centroid_v /= norm_v

        return IdentityNode(
            key=centroid_k.astype(np.float32),
            value=centroid_v.astype(np.float32),
            strength=total_s,
            children=list(l2_indices),
        )

    def _update_l3_centroid(self, identity: IdentityNode):
        """Recompute L3 centroid from L2 children."""
        valid = [i for i in identity.children if i < len(self.l2_patterns)]
        if not valid:
            return

        strengths = np.array([self.l2_patterns[i].strength for i in valid])
        keys = np.stack([self.l2_patterns[i].key for i in valid])
        values = np.stack([self.l2_patterns[i].value for i in valid])

        total_s = float(np.sum(strengths))
        if total_s < 1e-10:
            total_s = 1.0

        identity.key = (np.sum(strengths[:, None] * keys, axis=0) / total_s).astype(
            np.float32
        )
        identity.value = (np.sum(strengths[:, None] * values, axis=0) / total_s).astype(
            np.float32
        )
        identity.strength = total_s

        norm_k = np.linalg.norm(identity.key)
        if norm_k > 1e-8:
            identity.key /= norm_k
        norm_v = np.linalg.norm(identity.value)
        if norm_v > 1e-8:
            identity.value /= norm_v

    def hierarchical_recall(
        self,
        query_embedding: np.ndarray,
        memories: list[MemoryEntry],
        flat_results: list[tuple[MemoryEntry, float]],
    ) -> list[tuple[MemoryEntry, float]]:
        """Equation 23: Hierarchical recall with ResNet skip connections.

        R(q) = w₁·R_L1(q) + w₂·R_L2(q) + w₃·R_L3(q)

        L2 and L3 matches boost the scores of their child memories.
        Skip connections ensure all three layers contribute to every recall.

        Without hierarchy or with empty L2/L3, returns flat_results unchanged.
        """
        if not self.l2_patterns and not self.l3_identities:
            return flat_results
        if not flat_results:
            return flat_results

        # Build memory → index mapping for fast lookup
        mem_to_idx: dict[int, int] = {}
        for i, mem in enumerate(memories):
            mem_to_idx[id(mem)] = i

        # Score query against L2 patterns
        l2_boost: dict[int, float] = {}
        if self.l2_patterns:
            l2_keys = np.stack([p.key for p in self.l2_patterns])
            l2_sims = EmbeddingService.cosine_similarity_matrix(
                query_embedding, l2_keys
            ).astype(np.float64)

            for p_idx, pattern in enumerate(self.l2_patterns):
                l2_sim = max(0.0, float(l2_sims[p_idx]))
                if l2_sim > 0:
                    # Boost all children by L2 match score
                    for child_idx in pattern.children:
                        l2_boost[child_idx] = max(l2_boost.get(child_idx, 0.0), l2_sim)

        # Score query against L3 identities
        l3_boost: dict[int, float] = {}
        if self.l3_identities:
            l3_keys = np.stack([n.key for n in self.l3_identities])
            l3_sims = EmbeddingService.cosine_similarity_matrix(
                query_embedding, l3_keys
            ).astype(np.float64)

            for n_idx, identity in enumerate(self.l3_identities):
                l3_sim = max(0.0, float(l3_sims[n_idx]))
                if l3_sim > 0:
                    for l2_idx in identity.children:
                        if l2_idx < len(self.l2_patterns):
                            for child_idx in self.l2_patterns[l2_idx].children:
                                l3_boost[child_idx] = max(
                                    l3_boost.get(child_idx, 0.0), l3_sim
                                )

        # Apply skip connections: R = w1*R_L1 + w2*R_L2 + w3*R_L3
        w1 = self.config.recall_w1
        w2 = self.config.recall_w2
        w3 = self.config.recall_w3

        boosted = []
        for mem, score in flat_results:
            idx = mem_to_idx.get(id(mem), -1)
            l2_s = l2_boost.get(idx, 0.0)
            l3_s = l3_boost.get(idx, 0.0)

            # Skip connection: flat score + L2 boost + L3 boost
            new_score = w1 * score + w2 * l2_s * score + w3 * l3_s * score
            boosted.append((mem, new_score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def get_l2_summary(self, memories: list[MemoryEntry]) -> list[dict]:
        """Return L2 patterns with their children's texts for display."""
        summaries = []
        for p in self.l2_patterns:
            children_texts = []
            for idx in p.children:
                if idx < len(memories):
                    children_texts.append(memories[idx].input_text[:60])
            summaries.append(
                {
                    "strength": p.strength,
                    "n_children": len(p.children),
                    "children": children_texts,
                }
            )
        return summaries

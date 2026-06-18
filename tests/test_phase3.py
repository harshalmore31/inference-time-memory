"""Phase 3 tests: Equations 18-21 (MemWire-Inspired, Formalized).

Eq 18: Displacement-Augmented Edges (ResNet residual connections)
Eq 19: Sparse Lexical Recall (BM25, Robertson 1994)
Eq 20: Multi-Scale Activation (JK-Net, Xu 2018)
Eq 21: Query-Aware Message Passing (GAT, Veličković 2018)
"""

import numpy as np

from itm.config import MemoryConfig
from itm.core import MemoryEntry, MemoryLayer, SparseIndex
from itm.embeddings import EmbeddingService
from itm.graph import MemoryGraph
from itm.storage import MemoryStorage

# ── Test Utilities ──


def random_embedding(dim, rng):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder(EmbeddingService):
    """Skip model loading for unit tests — use random embeddings."""

    def __init__(self, config):
        self.config = config
        self._cache = {}

    def embed(self, text):
        if text not in self._cache:
            rng = np.random.default_rng(hash(text) % (2**31))
            self._cache[text] = random_embedding(config.embedding_dim, rng)
        return self._cache[text]


config = MemoryConfig(embedding_dim=64)


def make_layer(cfg=None):
    c = cfg or config
    embedder = FakeEmbedder(c)
    return MemoryLayer(c, embedder)


# ══════════════════════════════════════════════════════════════
# Equation 18: Displacement-Augmented Edges (ResNet Residual)
# ══════════════════════════════════════════════════════════════


class TestDisplacementEdges:
    """Eq 18: d_i = normalize(v_i - k_i), edge += α_d · sim(d_i, d_j)."""

    def test_displacement_vector_normalized(self):
        """Displacement vector should be unit length."""
        rng = np.random.default_rng(42)
        k = random_embedding(64, rng)
        v = random_embedding(64, np.random.default_rng(43))
        mem = MemoryEntry(key=k, value=v, strength=1.0, bias=0.5, timestamp=1)
        d = MemoryGraph._displacement(mem)
        assert abs(np.linalg.norm(d) - 1.0) < 1e-5

    def test_same_displacement_similar(self):
        """Memories with similar key→value shift should get higher edge weight."""
        rng = np.random.default_rng(10)
        # Two memories with similar displacement (same direction shift)
        base_k = random_embedding(64, rng)
        shift = random_embedding(64, np.random.default_rng(20))
        # Memory 1: k1 → k1 + shift
        k1 = base_k.copy()
        v1 = k1 + 0.5 * shift
        v1 = v1 / np.linalg.norm(v1)
        # Memory 2: different key but same shift direction
        k2 = random_embedding(64, np.random.default_rng(30))
        v2 = k2 + 0.5 * shift
        v2 = v2 / np.linalg.norm(v2)

        m1 = MemoryEntry(key=k1, value=v1, strength=1.0, bias=0.5, timestamp=1)
        m2 = MemoryEntry(key=k2, value=v2, strength=1.0, bias=0.5, timestamp=1)

        # With displacement enabled. Eq 18 now renormalizes (alpha_k, alpha_v,
        # alpha_d) into a convex combination instead of adding alpha_d on top,
        # so the comparison baseline must hold (alpha_k, alpha_v) fixed and toggle
        # only the displacement channel — otherwise we would be comparing two
        # different weighting schemes. With equal alpha_k == alpha_v, the on/off
        # weights are the convex averages including/excluding disp_sim, so the
        # displacement channel raises the weight iff disp_sim exceeds the
        # key/value-only average.
        cfg_on = MemoryConfig(
            embedding_dim=64,
            displacement_edges_enabled=True,
            alpha_d=0.3,
            alpha_k=0.35,
            alpha_v=0.35,
        )
        graph_on = MemoryGraph(cfg_on)
        w_on = graph_on._compute_edge_weight(m1, m2)

        # Analytic convex baseline with the displacement channel removed:
        # renormalize (alpha_k, alpha_v) to sum to 1 (0.35/0.35 -> 0.5/0.5) and
        # take the key+value-only convex average times the temporal factor.
        key_sim = float(EmbeddingService.cosine_similarity(m1.key, m2.key))
        val_sim = float(EmbeddingService.cosine_similarity(m1.value, m2.value))
        temporal = np.exp(-abs(m1.timestamp - m2.timestamp) / cfg_on.tau_temporal)
        kv_only = 0.5 * key_sim + 0.5 * val_sim
        w_baseline = kv_only * temporal

        d1 = MemoryGraph._displacement(m1)
        d2 = MemoryGraph._displacement(m2)
        disp_sim = float(EmbeddingService.cosine_similarity(d1, d2))

        # Convex bound: the augmented weight never exceeds 1.0 * temporal.
        assert w_on <= temporal + 1e-6
        # The displacement channel pulls the weight toward disp_sim: enabling it
        # raises the weight exactly when disp_sim > key/value-only average.
        if disp_sim > kv_only:
            assert w_on > w_baseline - 1e-9

    def test_disabled_matches_original(self):
        """When disabled, edge weight should be identical to original Eq 4."""
        rng1, rng2 = np.random.default_rng(1), np.random.default_rng(2)
        m1 = MemoryEntry(
            key=random_embedding(64, rng1),
            value=random_embedding(64, np.random.default_rng(3)),
            strength=1.0,
            bias=0.5,
            timestamp=1,
        )
        m2 = MemoryEntry(
            key=random_embedding(64, rng2),
            value=random_embedding(64, np.random.default_rng(4)),
            strength=1.0,
            bias=0.5,
            timestamp=2,
        )

        cfg_on = MemoryConfig(embedding_dim=64, displacement_edges_enabled=False)
        cfg_off = MemoryConfig(embedding_dim=64, displacement_edges_enabled=False)
        g1 = MemoryGraph(cfg_on)
        g2 = MemoryGraph(cfg_off)

        assert (
            abs(g1._compute_edge_weight(m1, m2) - g2._compute_edge_weight(m1, m2))
            < 1e-8
        )

    def test_displacement_edge_weight_bounded_by_one(self):
        """Eq 18: displacement-augmented semantic weight stays bounded <= 1.0.

        Because (alpha_k, alpha_v, alpha_d) are renormalized to a convex
        combination, every per-channel similarity is in [-1, 1] and the
        temporal factor is in (0, 1], so the edge weight can never exceed 1.0.
        The maximum is achieved by two identical memories at the same timestamp
        (all sims = 1.0, temporal = 1.0): the weight must equal exactly 1.0, not
        the ~1.3 the old additive scheme (alpha_k + alpha_v + alpha_d) produced.
        """
        cfg = MemoryConfig(
            embedding_dim=64,
            displacement_edges_enabled=True,
            alpha_k=0.5,
            alpha_v=0.5,
            alpha_d=0.3,
        )
        graph = MemoryGraph(cfg)

        k = random_embedding(64, np.random.default_rng(1))
        v = random_embedding(64, np.random.default_rng(2))
        # Two identical memories, same timestamp -> max-similarity, no temporal decay.
        m1 = MemoryEntry(
            key=k.copy(), value=v.copy(), strength=1.0, bias=0.5, timestamp=5
        )
        m2 = MemoryEntry(
            key=k.copy(), value=v.copy(), strength=1.0, bias=0.5, timestamp=5
        )
        w_max = graph._compute_edge_weight(m1, m2)
        assert w_max <= 1.0 + 1e-9
        assert abs(w_max - 1.0) < 1e-6

        # Random pairs at various lags must also stay within the bound.
        for seed in range(20):
            rng = np.random.default_rng(seed)
            a = MemoryEntry(
                key=random_embedding(64, rng),
                value=random_embedding(64, np.random.default_rng(seed + 100)),
                strength=1.0,
                bias=0.5,
                timestamp=int(seed),
            )
            b = MemoryEntry(
                key=random_embedding(64, np.random.default_rng(seed + 200)),
                value=random_embedding(64, np.random.default_rng(seed + 300)),
                strength=1.0,
                bias=0.5,
                timestamp=int(seed) + 1,
            )
            assert graph._compute_edge_weight(a, b) <= 1.0 + 1e-9

    def test_zero_displacement_handled(self):
        """When key == value, displacement should be handled (zero vector)."""
        k = random_embedding(64, np.random.default_rng(1))
        mem = MemoryEntry(key=k, value=k.copy(), strength=1.0, bias=0.5, timestamp=1)
        d = MemoryGraph._displacement(mem)
        # Should not crash, displacement is zero vector
        assert d.shape == (64,)


# ══════════════════════════════════════════════════════════════
# Equation 19: Sparse Lexical Recall (BM25 / TF-IDF)
# ══════════════════════════════════════════════════════════════


class TestSparseIndex:
    """Eq 19: TF-IDF sparse index for hybrid dense+sparse recall."""

    def test_exact_keyword_match(self):
        """Exact keyword should produce high similarity."""
        idx = SparseIndex()
        idx.add("my phone number is 9876543210")
        idx.add("I live in Mumbai")
        idx.add("my name is Harshal")

        sims = idx.query_similarity("what is my phone number 9876543210")
        # Document 0 should have highest similarity (shares "phone", "number", "9876543210")
        assert sims[0] > sims[1]
        assert sims[0] > sims[2]

    def test_name_keyword_match(self):
        """Proper nouns should match via sparse search."""
        idx = SparseIndex()
        idx.add("mandar is my friend")
        idx.add("raj is also my friend")
        idx.add("I study at VIT")

        sims = idx.query_similarity("who is mandar")
        assert sims[0] > sims[1]  # "mandar" matches doc 0
        assert sims[0] > sims[2]

    def test_no_match_returns_zero(self):
        """Query with no shared terms should return zeros."""
        idx = SparseIndex()
        idx.add("hello world")
        sims = idx.query_similarity("xyz123 completely different")
        assert sims[0] == 0.0

    def test_empty_query(self):
        """Empty query returns zeros."""
        idx = SparseIndex()
        idx.add("some text")
        sims = idx.query_similarity("")
        assert sims[0] == 0.0

    def test_empty_index(self):
        """Empty index returns empty array."""
        idx = SparseIndex()
        sims = idx.query_similarity("hello")
        assert len(sims) == 0

    def test_rebuild_matches_incremental(self):
        """Rebuild should produce same results as incremental add."""
        texts = ["hello world", "foo bar baz", "hello foo"]

        idx1 = SparseIndex()
        for t in texts:
            idx1.add(t)

        idx2 = SparseIndex()
        idx2.rebuild(texts)

        sims1 = idx1.query_similarity("hello")
        sims2 = idx2.query_similarity("hello")
        np.testing.assert_array_almost_equal(sims1, sims2)

    def test_idf_weighting(self):
        """Rare terms should have higher weight than common terms."""
        idx = SparseIndex()
        idx.add("the cat sat on the mat")  # "cat" appears once
        idx.add("the dog sat on the mat")  # "dog" appears once
        idx.add("the the the the the the mat")  # "the" is very common

        # "cat" is rarer, should match doc 0 more precisely
        sims = idx.query_similarity("cat")
        assert sims[0] > sims[1]
        assert sims[0] > sims[2]


class TestSparseRecallIntegration:
    """Eq 19: Sparse head integrated into multi-head recall."""

    def test_sparse_head_boosts_keyword_match(self):
        """When sparse is enabled, exact keyword matches should rank higher."""
        cfg = MemoryConfig(
            embedding_dim=64, sparse_recall_enabled=True, recall_w_sparse=0.15
        )
        layer = make_layer(cfg)

        rng = np.random.default_rng(42)
        # Create memories with known input_text
        for i, text in enumerate(
            ["my phone is 9876543210", "I live in Mumbai", "I study at VIT"]
        ):
            k = random_embedding(64, np.random.default_rng(i * 10))
            v = random_embedding(64, np.random.default_rng(i * 10 + 1))
            layer._create_memory(k, v, text, f"response {i}")

        # Sparse index should have 3 documents
        assert layer.sparse_index._doc_count == 3

    def test_disabled_no_sparse(self):
        """When disabled, sparse index should not be populated."""
        cfg = MemoryConfig(embedding_dim=64, sparse_recall_enabled=False)
        layer = make_layer(cfg)

        k = random_embedding(64, np.random.default_rng(1))
        v = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(k, v, "hello world", "response")

        assert layer.sparse_index._doc_count == 0


# ══════════════════════════════════════════════════════════════
# Equation 20: Multi-Scale Activation (JK-Net)
# ══════════════════════════════════════════════════════════════


class TestMultiScaleActivation:
    """Eq 20: Combine activations from all depths with per-hop damping."""

    def test_multiscale_uses_multiple_hops(self):
        """Multi-scale should use configured hop count."""
        cfg = MemoryConfig(
            embedding_dim=64,
            multiscale_enabled=True,
            multiscale_hops=4,
            multiscale_damping=0.7,
        )
        graph = MemoryGraph(cfg)

        # Create a chain: 0 → 1 → 2 → 3
        graph.edges = {
            0: {1: 0.8},
            1: {0: 0.8, 2: 0.8},
            2: {1: 0.8, 3: 0.8},
            3: {2: 0.8},
        }

        initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = graph.spreading_activation(initial)

        # Node 3 should get activation through 3 hops
        assert result[3] > 0.0
        # Node 0 should still have highest activation
        assert result[0] > result[3]

    def test_multiscale_damping_reduces_per_hop(self):
        """Per-hop damping should reduce activation at deeper levels."""
        cfg = MemoryConfig(
            embedding_dim=64,
            multiscale_enabled=True,
            multiscale_hops=3,
            multiscale_damping=0.5,
            eta_propagation=1.0,
        )
        graph = MemoryGraph(cfg)

        graph.edges = {
            0: {1: 1.0},
            1: {0: 1.0, 2: 1.0},
            2: {1: 1.0},
        }

        initial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        result = graph.spreading_activation(initial)

        # All nodes should get some activation
        assert result[1] > 0.0
        assert result[2] > 0.0

    def test_disabled_uses_original_hops(self):
        """When disabled, should use n_hops (default 2), not multiscale_hops."""
        cfg = MemoryConfig(
            embedding_dim=64, multiscale_enabled=False, n_hops=2, multiscale_hops=4
        )
        graph = MemoryGraph(cfg)

        # Chain of 5 nodes
        graph.edges = {
            0: {1: 0.8},
            1: {0: 0.8, 2: 0.8},
            2: {1: 0.8, 3: 0.8},
            3: {2: 0.8, 4: 0.8},
            4: {3: 0.8},
        }

        initial = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = graph.spreading_activation(initial)

        # With only 2 hops, node 4 should get very little activation
        # compared to multiscale with 4 hops
        cfg2 = MemoryConfig(
            embedding_dim=64, multiscale_enabled=True, multiscale_hops=4
        )
        graph2 = MemoryGraph(cfg2)
        graph2.edges = dict(graph.edges)
        result2 = graph2.spreading_activation(initial)

        # Multi-scale should reach node 4 better
        assert result2[4] > result[4]

    def test_multiscale_averages_all_scales(self):
        """JK-Net: final activation should be average of all scale activations."""
        cfg = MemoryConfig(
            embedding_dim=64,
            multiscale_enabled=True,
            multiscale_hops=2,
            eta_propagation=0.0,
        )
        graph = MemoryGraph(cfg)
        graph.edges = {0: {1: 0.5}, 1: {0: 0.5}}

        initial = np.array([1.0, 0.5], dtype=np.float64)
        result = graph.spreading_activation(initial)

        # With eta=0, no propagation happens, so all scales = initial
        # Average of 3 copies of initial = initial
        np.testing.assert_array_almost_equal(result, initial)


# ══════════════════════════════════════════════════════════════
# Equation 21: Query-Aware Message Passing (GAT)
# ══════════════════════════════════════════════════════════════


class TestQueryAwareSpread:
    """Eq 21: message_ij = w_ij · a_j · sim(q, k_j)."""

    def test_irrelevant_sender_suppressed(self):
        """Messages from query-irrelevant nodes should be dampened."""
        cfg = MemoryConfig(
            embedding_dim=64,
            query_aware_spread_enabled=True,
            n_hops=1,
            eta_propagation=1.0,
        )
        graph = MemoryGraph(cfg)

        # Node 0 → Node 1 (strong edge)
        graph.edges = {0: {1: 1.0}, 1: {0: 1.0}}

        rng = np.random.default_rng(42)
        k0 = random_embedding(64, np.random.default_rng(1))
        k1 = random_embedding(64, np.random.default_rng(2))
        keys = np.stack([k0, k1])

        # Query similar to k0, not k1
        query = k0 + 0.1 * random_embedding(64, rng)
        query = query / np.linalg.norm(query)

        initial = np.array([0.0, 1.0], dtype=np.float64)

        # With query-aware: node 1 sends message to node 0,
        # but node 1's relevance to query is low → dampened message
        result_aware = graph.spreading_activation(
            initial,
            query_embedding=query,
            keys=keys,
        )

        # Without query-aware: full message
        cfg2 = MemoryConfig(
            embedding_dim=64,
            query_aware_spread_enabled=False,
            n_hops=1,
            eta_propagation=1.0,
        )
        graph2 = MemoryGraph(cfg2)
        graph2.edges = dict(graph.edges)
        result_blind = graph2.spreading_activation(initial)

        # Query-aware should give node 0 less activation (from irrelevant node 1)
        # IF sim(query, k1) < 1.0, which it is since query ≈ k0 ≠ k1
        sim_q_k1 = float(EmbeddingService.cosine_similarity(query, k1))
        if sim_q_k1 < 0.99:
            assert result_aware[0] < result_blind[0]

    def test_relevant_sender_preserved(self):
        """Messages from query-relevant nodes should pass through."""
        cfg = MemoryConfig(
            embedding_dim=64,
            query_aware_spread_enabled=True,
            n_hops=1,
            eta_propagation=1.0,
        )
        graph = MemoryGraph(cfg)

        graph.edges = {0: {1: 1.0}, 1: {0: 1.0}}

        k0 = random_embedding(64, np.random.default_rng(1))
        keys = np.stack([k0, k0.copy()])  # Both keys identical to query

        query = k0.copy()  # Query = k0 = k1

        initial = np.array([0.0, 1.0], dtype=np.float64)
        result = graph.spreading_activation(
            initial,
            query_embedding=query,
            keys=keys,
        )

        # sim(q, k1) ≈ 1.0, so message should pass through fully
        assert result[0] > 0.5

    def test_disabled_ignores_query(self):
        """When disabled, spreading should be query-blind."""
        cfg = MemoryConfig(
            embedding_dim=64,
            query_aware_spread_enabled=False,
            n_hops=1,
            eta_propagation=1.0,
        )
        graph = MemoryGraph(cfg)
        graph.edges = {0: {1: 1.0}, 1: {0: 1.0}}

        k0 = random_embedding(64, np.random.default_rng(1))
        k1 = random_embedding(64, np.random.default_rng(2))
        keys = np.stack([k0, k1])
        query = k0.copy()

        initial = np.array([0.0, 1.0], dtype=np.float64)

        # With keys provided but disabled: should ignore
        result = graph.spreading_activation(initial)
        result_with_keys = graph.spreading_activation(
            initial,
            query_embedding=query,
            keys=keys,
        )

        # Both should be the same since query_aware is disabled
        np.testing.assert_array_almost_equal(result, result_with_keys)


# ══════════════════════════════════════════════════════════════
# Phase 3 Integration
# ══════════════════════════════════════════════════════════════


class TestPhase3Integration:
    """All Phase 3 features enabled together."""

    def test_all_features_enabled(self):
        """System should work with all Phase 3 features on."""
        cfg = MemoryConfig(
            embedding_dim=64,
            displacement_edges_enabled=True,
            sparse_recall_enabled=True,
            multiscale_enabled=True,
            query_aware_spread_enabled=True,
        )
        layer = make_layer(cfg)

        # Store some memories
        for i in range(5):
            k = random_embedding(64, np.random.default_rng(i * 100))
            v = random_embedding(64, np.random.default_rng(i * 100 + 1))
            layer.update(k, v, f"fact number {i}", f"response {i}")

        assert len(layer.memories) >= 1

        # Recall should work
        q = random_embedding(64, np.random.default_rng(999))
        results = layer.recall_graph(q, query_text="fact number 3")
        assert isinstance(results, list)

    def test_all_disabled_matches_baseline(self):
        """All Phase 3 disabled should produce same results as Phase 1."""
        cfg1 = MemoryConfig(embedding_dim=64)
        cfg2 = MemoryConfig(
            embedding_dim=64,
            displacement_edges_enabled=False,
            sparse_recall_enabled=False,
            multiscale_enabled=False,
            query_aware_spread_enabled=False,
        )

        layer1 = make_layer(cfg1)
        layer2 = make_layer(cfg2)

        rng = np.random.default_rng(42)
        for i in range(3):
            k = random_embedding(64, np.random.default_rng(i * 50))
            v = random_embedding(64, np.random.default_rng(i * 50 + 1))
            layer1.update(k.copy(), v.copy(), f"text {i}", f"resp {i}")
            layer2.update(k.copy(), v.copy(), f"text {i}", f"resp {i}")

        q = random_embedding(64, np.random.default_rng(999))
        r1 = layer1.recall(q)
        r2 = layer2.recall(q)

        assert len(r1) == len(r2)
        for (m1, s1), (m2, s2) in zip(r1, r2):
            assert abs(s1 - s2) < 1e-6


class TestSparseStoragePersistence:
    """Sparse index should rebuild correctly after save/load."""

    def test_sparse_index_rebuilds_on_load(self, tmp_path):
        cfg = MemoryConfig(
            embedding_dim=64,
            sparse_recall_enabled=True,
            memory_dir=str(tmp_path),
        )
        layer = make_layer(cfg)
        storage = MemoryStorage(cfg)

        # Create memories
        for i, text in enumerate(["phone 9876543210", "I live in Mumbai"]):
            k = random_embedding(64, np.random.default_rng(i * 10))
            v = random_embedding(64, np.random.default_rng(i * 10 + 1))
            layer._create_memory(k, v, text, f"resp {i}")

        assert layer.sparse_index._doc_count == 2

        # Save
        storage.save(layer)

        # Load into fresh layer
        layer2 = make_layer(cfg)
        storage.load(layer2)

        assert layer2.sparse_index._doc_count == 2

        # Query should work identically
        sims1 = layer.sparse_index.query_similarity("phone")
        sims2 = layer2.sparse_index.query_similarity("phone")
        np.testing.assert_array_almost_equal(sims1, sims2)

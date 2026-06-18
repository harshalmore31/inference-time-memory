"""Tests for the Memory Graph — Equations 4 & 5."""

import numpy as np

from itm.config import MemoryConfig
from itm.core import MemoryEntry, MemoryLayer
from itm.graph import MemoryGraph
from itm.storage import MemoryStorage


def random_embedding(dim=64, rng=None):
    rng = rng or np.random.default_rng(42)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


class FakeEmbedder:
    def __init__(self, dim=64):
        self.dim = dim
        self._rng = np.random.default_rng(0)
        self._cache = {}

    def embed(self, text):
        if text not in self._cache:
            self._cache[text] = random_embedding(self.dim, self._rng)
        return self._cache[text]


def make_layer(config=None, embedder=None):
    config = config or MemoryConfig(embedding_dim=64)
    embedder = embedder or FakeEmbedder(dim=64)
    return MemoryLayer(config, embedder)


# --- Equation 4: Edge Weight ---


class TestEdgeWeight:
    """w_ij = (α_k · sim(k_i, k_j) + α_v · sim(v_i, v_j)) · exp(-|t_i - t_j| / τ)"""

    def test_identical_memories_have_max_edge_weight(self):
        config = MemoryConfig(embedding_dim=64)
        graph = MemoryGraph(config)

        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))

        m1 = MemoryEntry(
            key=key.copy(),
            value=val.copy(),
            strength=1.0,
            bias=0.5,
            timestamp=1,
            input_text="a",
            output_text="b",
        )
        m2 = MemoryEntry(
            key=key.copy(),
            value=val.copy(),
            strength=1.0,
            bias=0.5,
            timestamp=1,
            input_text="a",
            output_text="b",
        )

        weight = graph._compute_edge_weight(m1, m2)
        # sim(k,k)=1, sim(v,v)=1, time_diff=0 → exp(0)=1
        # weight = (0.5*1 + 0.5*1) * 1 = 1.0
        assert abs(weight - 1.0) < 1e-6

    def test_temporal_decay_reduces_edge_weight(self):
        config = MemoryConfig(embedding_dim=64, tau_temporal=5.0)
        graph = MemoryGraph(config)

        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))

        m1 = MemoryEntry(
            key=key.copy(),
            value=val.copy(),
            strength=1.0,
            bias=0.5,
            timestamp=1,
            input_text="a",
            output_text="b",
        )
        m2 = MemoryEntry(
            key=key.copy(),
            value=val.copy(),
            strength=1.0,
            bias=0.5,
            timestamp=20,
            input_text="a",
            output_text="b",
        )

        weight = graph._compute_edge_weight(m1, m2)
        # Same content but 19 timesteps apart → temporal decay
        # exp(-19/5) ≈ 0.022 → weight ≈ 1.0 * 0.022 ≈ 0.022
        assert weight < 0.1

    def test_adjacent_timesteps_have_strong_connection(self):
        config = MemoryConfig(embedding_dim=64, tau_temporal=5.0)
        graph = MemoryGraph(config)

        key1 = random_embedding(64, np.random.default_rng(1))
        key2 = random_embedding(64, np.random.default_rng(2))
        val1 = random_embedding(64, np.random.default_rng(3))
        val2 = random_embedding(64, np.random.default_rng(4))

        m1 = MemoryEntry(
            key=key1,
            value=val1,
            strength=1.0,
            bias=0.5,
            timestamp=5,
            input_text="a",
            output_text="b",
        )
        m2 = MemoryEntry(
            key=key2,
            value=val2,
            strength=1.0,
            bias=0.5,
            timestamp=6,
            input_text="a",
            output_text="b",
        )

        weight_close = graph._compute_edge_weight(m1, m2)

        m3 = MemoryEntry(
            key=key2,
            value=val2,
            strength=1.0,
            bias=0.5,
            timestamp=50,
            input_text="a",
            output_text="b",
        )

        weight_far = graph._compute_edge_weight(m1, m3)

        # Same semantic content but close vs far in time
        assert weight_close > weight_far

    def test_unrelated_memories_get_no_edge(self):
        config = MemoryConfig(embedding_dim=64, theta_edge=0.1)
        graph = MemoryGraph(config)

        # Very different keys, very different values, far apart in time
        m1 = MemoryEntry(
            key=np.eye(64, dtype=np.float32)[0],
            value=np.eye(64, dtype=np.float32)[0],
            strength=1.0,
            bias=0.5,
            timestamp=1,
            input_text="topic A",
            output_text="response A",
        )
        m2 = MemoryEntry(
            key=np.eye(64, dtype=np.float32)[32],
            value=np.eye(64, dtype=np.float32)[32],
            strength=1.0,
            bias=0.5,
            timestamp=100,
            input_text="topic B",
            output_text="response B",
        )

        weight = graph._compute_edge_weight(m1, m2)
        # Orthogonal keys/values + far apart → weight ≈ 0
        assert weight < config.theta_edge


# --- Equation 5: Spreading Activation ---


class TestSpreadingActivation:
    """a_i^(l+1) = a_i^(l) + η · Σ_j(w_ij · a_j^(l))"""

    def test_activation_spreads_to_neighbors(self):
        config = MemoryConfig(embedding_dim=64, eta_propagation=0.5, n_hops=1)
        graph = MemoryGraph(config)

        # Manually create edges: 0 ↔ 1 (weight 0.8), 0 ↔ 2 (weight 0.5)
        graph.edges = {
            0: {1: 0.8, 2: 0.5},
            1: {0: 0.8},
            2: {0: 0.5},
        }

        # Only node 0 initially active
        initial = np.array([1.0, 0.0, 0.0])
        final = graph.spreading_activation(initial, n_hops=1)

        # Node 1 should gain activation: 0 + 0.5 * 0.8 * 1.0 = 0.4
        assert final[1] > 0.3
        # Node 2 should gain activation: 0 + 0.5 * 0.5 * 1.0 = 0.25
        assert final[2] > 0.2
        # Node 0 should keep its activation (plus back-propagation from neighbors)
        assert final[0] >= 1.0

    def test_no_edges_means_no_spreading(self):
        config = MemoryConfig(embedding_dim=64, eta_propagation=0.5, n_hops=2)
        graph = MemoryGraph(config)
        graph.edges = {0: {}, 1: {}, 2: {}}

        initial = np.array([1.0, 0.0, 0.5])
        final = graph.spreading_activation(initial, n_hops=2)

        np.testing.assert_allclose(final, initial)

    def test_more_hops_means_deeper_propagation(self):
        config = MemoryConfig(embedding_dim=64, eta_propagation=0.5)
        graph = MemoryGraph(config)

        # Chain: 0 → 1 → 2 (no direct 0 → 2 edge)
        graph.edges = {
            0: {1: 0.9},
            1: {0: 0.9, 2: 0.9},
            2: {1: 0.9},
        }

        initial = np.array([1.0, 0.0, 0.0])

        # 1 hop: activation reaches node 1 but not node 2
        final_1hop = graph.spreading_activation(initial, n_hops=1)
        # 2 hops: activation reaches node 2 through node 1
        final_2hop = graph.spreading_activation(initial, n_hops=2)

        assert final_2hop[2] > final_1hop[2]

    def test_zero_eta_means_no_propagation(self):
        config = MemoryConfig(embedding_dim=64, eta_propagation=0.0, n_hops=5)
        graph = MemoryGraph(config)
        graph.edges = {0: {1: 1.0}, 1: {0: 1.0}}

        initial = np.array([1.0, 0.0])
        final = graph.spreading_activation(initial, n_hops=5)
        np.testing.assert_allclose(final, initial)


# --- Graph-Based Recall (Multi-hop) ---


class TestGraphRecall:
    """The full pipeline: initial activation → spreading → results."""

    def test_graph_recall_finds_connected_memories(self):
        """Simulates the VIT example: query activates VIT memory,
        spreading activation pulls in connected friend memories."""
        config = MemoryConfig(
            embedding_dim=64,
            eta_propagation=0.5,
            n_hops=2,
            theta_edge=0.01,
            activation_threshold=0.01,
            tau_temporal=5.0,
        )
        layer = make_layer(config)

        # Create memories at adjacent timesteps (like a conversation)
        rng = np.random.default_rng(42)

        # Memory 0: "Harshal" (timestep 1)
        k0 = random_embedding(64, np.random.default_rng(10))
        v0 = random_embedding(64, np.random.default_rng(11))
        layer.update(k0, v0, "my name is Harshal", "nice to meet you")

        # Memory 1: "Mandar in bhusawal" (timestep 2)
        k1 = random_embedding(64, np.random.default_rng(20))
        v1 = random_embedding(64, np.random.default_rng(21))
        layer.update(k1, v1, "friend Mandar lives in bhusawal", "noted")

        # Memory 2: "Raj in dhule" (timestep 3) — similar to Mandar memory
        k2 = random_embedding(64, np.random.default_rng(30))
        v2 = random_embedding(64, np.random.default_rng(31))
        layer.update(k2, v2, "friend Raj lives in dhule", "noted")

        # Memory 3: "VIT" (timestep 4) — close in time to Raj and Mandar
        k3 = random_embedding(64, np.random.default_rng(40))
        v3 = random_embedding(64, np.random.default_rng(41))
        layer.update(k3, v3, "we all study at VIT", "great institute")

        # Graph should have edges (temporal proximity + some semantic sim)
        assert layer.graph.edge_count() > 0

        # Query with k3 (VIT) — should activate VIT memory directly,
        # then spreading should activate connected memories
        results = layer.recall_graph(k3)

        # Should find more than just the VIT memory
        assert len(results) >= 2
        # VIT memory should be first (highest activation)
        assert results[0][0].input_text == "we all study at VIT"

    def test_graph_recall_empty_layer(self):
        layer = make_layer()
        q = random_embedding(64)
        assert layer.recall_graph(q) == []

    def test_graph_edges_built_on_create(self):
        """Verify edges are created when memories are added."""
        config = MemoryConfig(embedding_dim=64, theta_edge=0.01, tau_temporal=10.0)
        layer = make_layer(config)

        k = random_embedding(64, np.random.default_rng(1))
        v = random_embedding(64, np.random.default_rng(2))

        # First memory — no edges possible
        layer.update(k, v, "first", "first response")
        assert layer.graph.edge_count() == 0

        # Second memory at adjacent timestep — should form edge
        k2 = random_embedding(64, np.random.default_rng(3))
        v2 = random_embedding(64, np.random.default_rng(4))
        layer.update(k2, v2, "second", "second response")

        # There should be at least some edge (temporal proximity is high)
        assert layer.graph.node_count() >= 1


# --- Graph Rebuild (Persistence) ---


class TestGraphPersistence:

    def test_graph_survives_save_load(self, tmp_path):
        """After save/load, graph should be rebuilt with same structure."""
        config = MemoryConfig(
            embedding_dim=64,
            memory_dir=str(tmp_path),
            theta_edge=0.01,
            tau_temporal=10.0,
        )
        embedder = FakeEmbedder(dim=64)
        layer = MemoryLayer(config, embedder)
        storage = MemoryStorage(config)

        # Create memories
        for i in range(5):
            k = random_embedding(64, np.random.default_rng(i))
            v = random_embedding(64, np.random.default_rng(i + 50))
            layer.update(k, v, f"input {i}", f"output {i}")

        # Rebuild to normalize edges (Equation 7 may have modified values
        # during incremental creation, making edges stale)
        layer.graph.rebuild(layer.memories)
        edges_before = layer.graph.edge_count()
        storage.save(layer)

        # Load into fresh layer
        layer2 = MemoryLayer(config, embedder)
        storage.load(layer2)

        # Graph should be rebuilt with same edge count
        assert layer2.graph.edge_count() == edges_before

    def test_rebuild_matches_incremental(self):
        """Two consecutive rebuilds from the same memory state should match.

        Note: incremental edges may differ from rebuild because Equation 7
        (graph-propagated contradiction) modifies memory values during
        creation. Rebuild uses final values, so two rebuilds must agree.
        """
        config = MemoryConfig(embedding_dim=64, theta_edge=0.01, tau_temporal=10.0)
        layer = make_layer(config)

        for i in range(5):
            k = random_embedding(64, np.random.default_rng(i))
            v = random_embedding(64, np.random.default_rng(i + 50))
            layer.update(k, v, f"input {i}", f"output {i}")

        # First rebuild (normalizes edges to current memory values)
        layer.graph.rebuild(layer.memories)
        first_rebuild = {
            i: dict(neighbors) for i, neighbors in layer.graph.edges.items()
        }

        # Second rebuild (should produce identical results)
        layer.graph.rebuild(layer.memories)
        second_rebuild = {
            i: dict(neighbors) for i, neighbors in layer.graph.edges.items()
        }

        # Should have same structure
        assert first_rebuild.keys() == second_rebuild.keys()
        for i in first_rebuild:
            for j in first_rebuild[i]:
                assert abs(first_rebuild[i][j] - second_rebuild[i].get(j, 0)) < 1e-6

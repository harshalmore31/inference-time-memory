"""Tests for Memory Input Gate (Equation 6) & Graph-Propagated Contradiction (Equation 7)."""

import numpy as np

from itm.config import MemoryConfig
from itm.core import MemoryEntry, MemoryLayer
from itm.embeddings import EmbeddingService
from itm.graph import MemoryGraph


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


# --- Equation 6: Memory Input Gate ---


class TestMemoryInputGate:
    """g = σ(β_g · ((1 - R_overlap) - θ_gate))"""

    def test_low_overlap_opens_gate(self):
        """When output doesn't echo stored knowledge → gate opens → store."""
        config = MemoryConfig(embedding_dim=64, beta_gate=8.0, theta_gate=0.5)
        layer = make_layer(config)

        # Low R_overlap = new information (fact)
        gate = layer._memory_input_gate(R_overlap=0.1)
        # (1 - 0.1) - 0.5 = 0.4 → σ(8 * 0.4) = σ(3.2) ≈ 0.96
        assert gate > 0.9

    def test_high_overlap_closes_gate(self):
        """When output echoes stored knowledge → gate closes → skip."""
        config = MemoryConfig(embedding_dim=64, beta_gate=8.0, theta_gate=0.5)
        layer = make_layer(config)

        # High R_overlap = retrieval (query response)
        gate = layer._memory_input_gate(R_overlap=0.85)
        # (1 - 0.85) - 0.5 = -0.35 → σ(8 * -0.35) = σ(-2.8) ≈ 0.06
        assert gate < 0.1

    def test_borderline_overlap_gives_moderate_gate(self):
        """At the threshold center, gate ≈ 0.5."""
        config = MemoryConfig(embedding_dim=64, beta_gate=8.0, theta_gate=0.5)
        layer = make_layer(config)

        # R_overlap = 0.5 → (1 - 0.5) - 0.5 = 0.0 → σ(0) = 0.5
        gate = layer._memory_input_gate(R_overlap=0.5)
        assert abs(gate - 0.5) < 0.01

    def test_best_sim_overrides_R_overlap_as_primary(self):
        """When best_sim is provided, it becomes the primary signal instead of R_overlap."""
        config = MemoryConfig(
            embedding_dim=64,
            beta_gate=6.0,
            theta_gate=0.4,
            gate_w_out=0.10,
        )
        layer = make_layer(config)

        # Case 1: low best_sim (novel input) → gate opens even with high R_out
        gate_novel = layer._memory_input_gate(
            R_overlap=0.8,
            best_sim=0.2,  # R_out high but input is novel
        )
        # primary = 1 - 0.2 = 0.8, penalty = 0.10*0.8 = 0.08, novelty = 0.72
        assert gate_novel > 0.8

        # Case 2: high best_sim (related query) → gate closes even with low R_out
        gate_query = layer._memory_input_gate(
            R_overlap=0.3,
            best_sim=0.85,  # R_out low but input matches stored key
        )
        # primary = 1 - 0.85 = 0.15, penalty = 0.10*0.3 = 0.03, novelty = 0.12
        assert gate_query < 0.3

    def test_retrieval_overlap_empty_memory_is_zero(self):
        """With no memories, R_overlap = 0 (nothing to echo)."""
        config = MemoryConfig(embedding_dim=64)
        layer = make_layer(config)

        output_emb = random_embedding(64, np.random.default_rng(1))
        R_overlap = layer._compute_retrieval_overlap(output_emb)
        assert R_overlap == 0.0

    def test_retrieval_overlap_with_similar_output(self):
        """Output similar to a stored key → high R_overlap."""
        config = MemoryConfig(embedding_dim=64)
        layer = make_layer(config)

        # Create a memory with a known key
        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key, val, "test input", "test output")

        # Output embedding = same as the key (simulates LLM echoing stored info)
        R_overlap = layer._compute_retrieval_overlap(key)
        assert R_overlap > 0.99

    def test_retrieval_overlap_with_dissimilar_output(self):
        """Output different from stored keys → low R_overlap."""
        config = MemoryConfig(embedding_dim=64)
        layer = make_layer(config)

        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key, val, "test input", "test output")

        # Output embedding is orthogonal to the key
        orthogonal = np.zeros(64, dtype=np.float32)
        orthogonal[32] = 1.0  # basis vector far from random key
        R_overlap = layer._compute_retrieval_overlap(orthogonal)
        assert R_overlap < 0.5


class TestGateInUpdateFlow:
    """Gate integrated into MemoryLayer.update() decision flow."""

    def test_query_response_is_filtered(self):
        """When query input is related to stored key AND output echoes it → filtered.

        Simulates: "what is my name?" vs stored "my name is Harshal".
        In real BGE-M3, these have high similarity. We simulate by making
        the query input a perturbation of the stored key (best_sim ≈ 0.8+).
        """
        config = MemoryConfig(
            embedding_dim=64,
            beta_gate=8.0,
            theta_gate=0.5,
            theta_min_gate=0.1,
            gate_w_out=0.10,
            gate_w_val=0.10,
            gate_w_ctx=0.10,
        )
        layer = make_layer(config)

        # Create a memory: "my name is Harshal"
        key = random_embedding(64, np.random.default_rng(10))
        val = random_embedding(64, np.random.default_rng(11))
        layer._create_memory(key, val, "my name is Harshal", "nice to meet you")
        layer.timestep = 1

        # Query input is semantically RELATED to the stored key but below theta_create.
        # 0.50*key + 0.50*noise → best_sim ≈ 0.71 (below theta_create=0.75, so gate path)
        # In real BGE-M3, "what is my name?" vs "my name is Harshal" ≈ 0.6-0.7.
        noise = random_embedding(64, np.random.default_rng(20))
        query_input = 0.50 * key + 0.50 * noise
        query_input = (query_input / np.linalg.norm(query_input)).astype(np.float32)

        # Output also echoes the stored key (LLM retrieves the answer)
        query_output = key.copy()

        result = layer.update(
            query_input, query_output, "what is my name", "Your name is Harshal"
        )
        assert result["action"] == "filtered"
        assert len(layer.memories) == 1  # No new memory created

    def test_fact_response_is_stored(self):
        """When LLM output doesn't echo stored keys, the input should be stored."""
        config = MemoryConfig(
            embedding_dim=64,
            beta_gate=8.0,
            theta_gate=0.5,
            theta_min_gate=0.1,
        )
        layer = make_layer(config)

        # Create a memory
        key = random_embedding(64, np.random.default_rng(10))
        val = random_embedding(64, np.random.default_rng(11))
        layer._create_memory(key, val, "my name is Harshal", "nice to meet you")
        layer.timestep = 1

        # New fact: input is different, output is also different from stored keys
        fact_input = random_embedding(
            64, np.random.default_rng(30)
        )  # "I live in Mumbai"
        fact_output = random_embedding(64, np.random.default_rng(31))  # "Got it!"

        result = layer.update(fact_input, fact_output, "I live in Mumbai", "Got it!")
        assert result["action"] == "created"
        assert len(layer.memories) == 2  # New memory created

    def test_gated_strength_scales_initial_strength(self):
        """New memory strength = gate * initial_strength."""
        config = MemoryConfig(
            embedding_dim=64,
            beta_gate=8.0,
            theta_gate=0.5,
            theta_min_gate=0.1,
            initial_strength=1.0,
        )
        layer = make_layer(config)

        # First memory (no gate applied since memory is empty)
        key1 = random_embedding(64, np.random.default_rng(1))
        val1 = random_embedding(64, np.random.default_rng(2))
        layer.update(key1, val1, "first fact", "ok")

        # Second memory: output moderately overlaps with first key
        key2 = random_embedding(64, np.random.default_rng(3))
        # Create output that has some overlap with key1 (mix key1 + noise)
        noise = random_embedding(64, np.random.default_rng(4))
        partial_overlap = 0.5 * key1 + 0.5 * noise
        partial_overlap = partial_overlap / np.linalg.norm(partial_overlap)

        result = layer.update(key2, partial_overlap, "second fact", "response")
        if result["action"] == "created":
            new_mem = layer.memories[-1]
            # Strength should be gate * initial_strength, which is < initial_strength
            # because output has some overlap
            assert new_mem.strength <= config.initial_strength

    def test_first_memory_always_created(self):
        """First memory bypasses gate (no stored keys to compare against)."""
        config = MemoryConfig(embedding_dim=64)
        layer = make_layer(config)

        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))
        result = layer.update(key, val, "first memory", "acknowledged")

        assert result["action"] == "created"
        assert len(layer.memories) == 1

    def test_novel_input_stored_despite_high_R_out(self):
        """New fact about novel entity should be stored even when LLM output
        echoes stored knowledge (the Sanika case).

        Simulates: "sanika is project partner" when stored keys are about
        om/ashish. Input is novel (low best_sim), but output mentions
        existing partners (high R_out). Gate should OPEN because primary
        signal (input novelty) is strong.
        """
        config = MemoryConfig(
            embedding_dim=64,
            beta_gate=6.0,
            theta_gate=0.4,
            theta_min_gate=0.1,
            gate_w_out=0.10,
            gate_w_val=0.10,
            gate_w_ctx=0.10,
        )
        layer = make_layer(config)

        # Create a memory about om/ashish
        key_existing = random_embedding(64, np.random.default_rng(10))
        val_existing = random_embedding(64, np.random.default_rng(11))
        layer._create_memory(
            key_existing, val_existing, "om and ashish are project partners", "got it"
        )
        layer.timestep = 1

        # Novel input about Sanika (unrelated to stored key → low best_sim)
        novel_input = random_embedding(64, np.random.default_rng(30))

        # But output echoes stored knowledge (LLM mentions om/ashish → R_out ≈ 0.71)
        # In real BGE-M3, Sanika's R_out is 0.800 — below theta_out_filter (0.90).
        noisy_echo = 0.5 * key_existing + 0.5 * random_embedding(
            64, np.random.default_rng(31)
        )
        noisy_echo = (noisy_echo / np.linalg.norm(noisy_echo)).astype(np.float32)

        result = layer.update(
            novel_input,
            noisy_echo,
            "sanika is also project partner",
            "I'll remember sanika alongside om and ashish",
        )
        assert result["action"] == "created"
        assert len(layer.memories) == 2

    def test_high_R_out_ceiling_filters_retrieval(self):
        """When R_out >= theta_out_filter (0.90), always filter regardless of gate value.

        Simulates: "what is my name?" where LLM output is 95% similar to stored key.
        Even with moderate best_sim, the R_out ceiling catches clear retrieval.
        """
        config = MemoryConfig(
            embedding_dim=64,
            theta_out_filter=0.90,
            beta_gate=6.0,
            theta_gate=0.4,
            theta_min_gate=0.1,
        )
        layer = make_layer(config)

        # Store a memory
        key = random_embedding(64, np.random.default_rng(10))
        val = random_embedding(64, np.random.default_rng(11))
        layer._create_memory(key, val, "my name is Harshal", "nice to meet you")
        layer.timestep = 1

        # Novel input (low best_sim) but output is almost identical to stored key
        novel_input = random_embedding(64, np.random.default_rng(50))
        retrieval_output = 0.95 * key + 0.05 * random_embedding(
            64, np.random.default_rng(51)
        )
        retrieval_output = (retrieval_output / np.linalg.norm(retrieval_output)).astype(
            np.float32
        )

        result = layer.update(
            novel_input, retrieval_output, "what is my name", "Your name is Harshal"
        )
        assert result["action"] == "filtered"
        assert "retrieval ceiling" in result["details"]
        assert len(layer.memories) == 1

    def test_strengthen_path_unchanged_by_gate(self):
        """Gate only affects CREATE path. Strengthen path is unchanged."""
        config = MemoryConfig(embedding_dim=64, theta_create=0.75)
        layer = make_layer(config)

        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key, val, "Python is great", "agreed!")
        layer.timestep = 1

        original_strength = layer.memories[0].strength

        # Same key (sim > theta_create) → strengthen regardless of output
        result = layer.update(key.copy(), val.copy(), "Python is great", "agreed!")
        assert result["action"] == "strengthened"
        assert layer.memories[0].strength > original_strength


# --- Equation 7: Graph-Propagated Contradiction ---


class TestGraphPropagatedContradiction:
    """For each neighbor j: if val_sim < θ_value: apply Eq 2 through graph."""

    def test_contradiction_propagates_to_graph_neighbor(self):
        """New memory contradicts a graph neighbor → neighbor value shifts + weakened."""
        config = MemoryConfig(
            embedding_dim=64,
            theta_edge=0.01,
            tau_temporal=10.0,
            theta_value=0.5,
            delta_graph=0.7,
        )
        graph = MemoryGraph(config)

        # Create two memories with SIMILAR keys (will form edge) but DIFFERENT values
        key_shared = random_embedding(64, np.random.default_rng(1))
        val_old = random_embedding(64, np.random.default_rng(10))
        val_new = random_embedding(64, np.random.default_rng(20))

        mem_old = MemoryEntry(
            key=key_shared.copy(),
            value=val_old.copy(),
            strength=1.0,
            bias=0.5,
            timestamp=1,
            input_text="old fact",
            output_text="old response",
        )
        mem_new = MemoryEntry(
            key=key_shared.copy(),
            value=val_new.copy(),
            strength=1.0,
            bias=0.5,
            timestamp=2,
            input_text="new fact",
            output_text="new response",
        )
        memories = [mem_old, mem_new]

        # Build edges
        graph.add_memory(0, memories[:1])
        graph.add_memory(1, memories)

        original_strength = mem_old.strength
        original_value = mem_old.value.copy()

        # Propagate contradiction from mem_new (index 1)
        corrected = graph.propagate_contradiction(1, memories, config)

        # Check that val_sim between different random vectors is < theta_value
        val_sim = EmbeddingService.cosine_similarity(val_old, val_new)
        if val_sim < config.theta_value:
            assert 0 in corrected
            assert mem_old.strength < original_strength
            assert mem_old.strength == original_strength * config.delta_graph
            # Value should have shifted toward new value
            assert not np.allclose(mem_old.value, original_value)

    def test_no_contradiction_when_values_agree(self):
        """If neighbor values are similar, no contradiction propagates."""
        config = MemoryConfig(
            embedding_dim=64,
            theta_edge=0.01,
            tau_temporal=10.0,
            theta_value=0.5,
            delta_graph=0.7,
        )
        graph = MemoryGraph(config)

        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))

        # Same value → no contradiction
        mem1 = MemoryEntry(
            key=key.copy(),
            value=val.copy(),
            strength=1.0,
            bias=0.5,
            timestamp=1,
            input_text="a",
            output_text="b",
        )
        mem2 = MemoryEntry(
            key=key.copy(),
            value=val.copy(),
            strength=1.0,
            bias=0.5,
            timestamp=2,
            input_text="a",
            output_text="b",
        )
        memories = [mem1, mem2]

        graph.add_memory(0, memories[:1])
        graph.add_memory(1, memories)

        corrected = graph.propagate_contradiction(1, memories, config)
        assert len(corrected) == 0
        assert mem1.strength == 1.0  # Unchanged

    def test_contradiction_propagates_through_chain(self):
        """Contradiction in _create_memory propagates when edges exist."""
        config = MemoryConfig(
            embedding_dim=64,
            theta_edge=0.01,
            tau_temporal=10.0,
            theta_value=0.5,
            delta_graph=0.7,
        )
        layer = make_layer(config)

        # Memory 0: original fact
        k0 = random_embedding(64, np.random.default_rng(1))
        v0 = random_embedding(64, np.random.default_rng(10))
        layer.update(k0, v0, "om is my friend", "ok")

        original_strength = layer.memories[0].strength

        # Memory 1: contradicting fact (similar key, different value, close in time)
        k1 = k0 * 0.9 + random_embedding(64, np.random.default_rng(2)) * 0.1
        k1 = k1 / np.linalg.norm(k1)
        v1 = random_embedding(64, np.random.default_rng(20))

        # Use a very different output so gate opens
        layer.update(k1, v1, "om is my partner not friend", "noted")

        # If contradiction was propagated, memory 0 should be weakened
        val_sim = EmbeddingService.cosine_similarity(v0, v1)
        if val_sim < config.theta_value and layer.graph.edge_count() > 0:
            assert layer.memories[0].strength < original_strength

    def test_no_propagation_with_no_edges(self):
        """If new memory has no graph edges, no contradiction propagates."""
        config = MemoryConfig(
            embedding_dim=64,
            theta_edge=0.99,  # Very high threshold → no edges
        )
        graph = MemoryGraph(config)
        graph.edges = {0: {}, 1: {}}

        mem1 = MemoryEntry(
            key=random_embedding(64, np.random.default_rng(1)),
            value=random_embedding(64, np.random.default_rng(2)),
            strength=1.0,
            bias=0.5,
            timestamp=1,
            input_text="a",
            output_text="b",
        )
        mem2 = MemoryEntry(
            key=random_embedding(64, np.random.default_rng(3)),
            value=random_embedding(64, np.random.default_rng(4)),
            strength=1.0,
            bias=0.5,
            timestamp=2,
            input_text="c",
            output_text="d",
        )
        memories = [mem1, mem2]

        corrected = graph.propagate_contradiction(1, memories, config)
        assert len(corrected) == 0
        assert mem1.strength == 1.0


# --- Equation 2b: Soft Value Drift on Strengthen ---


class TestSoftValueDrift:
    """v_i += b_i · (1 - val_sim) · (v_new - v_i) — always-on proportional nudge."""

    def test_strengthen_applies_value_drift(self):
        """On strengthen (no contradiction), value should drift toward new output."""
        config = MemoryConfig(embedding_dim=64, theta_create=0.75, initial_bias=0.5)
        layer = make_layer(config)

        # Create memory with known value
        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key, val, "my friend mandar", "ok")
        original_value = layer.memories[0].value.copy()
        layer.timestep = 1

        # Strengthen with same key but output value similar enough to avoid contradiction
        # (val_sim >= theta_value=0.5). Use perturbation of original value.
        noise = random_embedding(64, np.random.default_rng(3))
        new_val = 0.8 * val + 0.2 * noise
        new_val = (new_val / np.linalg.norm(new_val)).astype(np.float32)
        result = layer.update(key.copy(), new_val, "my friend mandar", "got it")

        assert result["action"] == "strengthened"
        assert "val_drift" in result["details"]
        # Value should have shifted (not identical to original)
        assert not np.allclose(layer.memories[0].value, original_value, atol=1e-4)

    def test_drift_proportional_to_dissimilarity(self):
        """Larger value dissimilarity → larger drift magnitude."""
        config = MemoryConfig(embedding_dim=64, theta_create=0.75, initial_bias=0.5)

        # Case 1: very similar output → small drift
        layer1 = make_layer(config)
        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))
        layer1._create_memory(key, val, "test", "ok")
        orig1 = layer1.memories[0].value.copy()
        layer1.timestep = 1
        # Similar output (small perturbation of val)
        similar_val = 0.95 * val + 0.05 * random_embedding(64, np.random.default_rng(5))
        similar_val = (similar_val / np.linalg.norm(similar_val)).astype(np.float32)
        layer1.update(key.copy(), similar_val, "test", "ok")
        drift1 = np.linalg.norm(layer1.memories[0].value - orig1)

        # Case 2: very different output → large drift
        layer2 = make_layer(config)
        layer2._create_memory(key.copy(), val.copy(), "test", "ok")
        orig2 = layer2.memories[0].value.copy()
        layer2.timestep = 1
        diff_val = random_embedding(64, np.random.default_rng(10))
        layer2.update(key.copy(), diff_val, "test", "ok")
        drift2 = np.linalg.norm(layer2.memories[0].value - orig2)

        # Different output should cause larger drift
        assert drift2 > drift1

    def test_drift_preserves_unit_norm(self):
        """After drift, value vector should remain unit-normalized."""
        config = MemoryConfig(embedding_dim=64, theta_create=0.75, initial_bias=0.5)
        layer = make_layer(config)

        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key, val, "test", "ok")
        layer.timestep = 1

        # Use value similar enough to avoid contradiction (val_sim >= 0.5)
        noise = random_embedding(64, np.random.default_rng(3))
        new_val = 0.8 * val + 0.2 * noise
        new_val = (new_val / np.linalg.norm(new_val)).astype(np.float32)
        result = layer.update(key.copy(), new_val, "test", "ok")
        assert result["action"] == "strengthened"

        norm = np.linalg.norm(layer.memories[0].value)
        assert abs(norm - 1.0) < 1e-5


# --- Equation 5b: Degree-Normalized Spreading Activation ---


class TestDegreeNormalizedSpreading:
    """a_i^(l+1) = a_i^(l) + η · (1/|N(i)|) · Σ_j(w_ij · a_j^(l))"""

    def test_mean_aggregation_normalizes_by_degree(self):
        """A high-degree node should get the same boost magnitude as low-degree."""
        config = MemoryConfig(embedding_dim=64, eta_propagation=1.0, n_hops=1)
        graph = MemoryGraph(config)

        # Node 0: 1 neighbor (node 1, weight 0.5)
        # Node 1: 3 neighbors (nodes 0, 2, 3, all weight 0.5)
        graph.edges = {
            0: {1: 0.5},
            1: {0: 0.5, 2: 0.5, 3: 0.5},
            2: {1: 0.5},
            3: {1: 0.5},
        }

        # All nodes start with activation 1.0
        initial = np.array([1.0, 1.0, 1.0, 1.0])
        result = graph.spreading_activation(initial, n_hops=1)

        # Node 0 (1 neighbor): delta = 0.5 * 1.0 / 1 = 0.5
        # Node 1 (3 neighbors): delta = (0.5*1.0 + 0.5*1.0 + 0.5*1.0) / 3 = 0.5
        # Both should get the same delta (mean normalization)
        delta_0 = result[0] - initial[0]
        delta_1 = result[1] - initial[1]
        assert abs(delta_0 - delta_1) < 1e-10

    def test_high_degree_doesnt_dominate(self):
        """Without normalization, hub nodes would get disproportionate activation.
        With mean aggregation, they should be comparable to leaf nodes."""
        config = MemoryConfig(embedding_dim=64, eta_propagation=0.5, n_hops=2)
        graph = MemoryGraph(config)

        # Star topology: node 0 is hub with 5 neighbors
        graph.edges = {
            0: {1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4, 5: 0.4},
            1: {0: 0.4},
            2: {0: 0.4},
            3: {0: 0.4},
            4: {0: 0.4},
            5: {0: 0.4},
        }

        # Only hub is initially activated
        initial = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = graph.spreading_activation(initial, n_hops=2)

        # Hub's activation should NOT explode due to many connections
        # With mean agg, hub boost = eta * mean(neighbors) per hop
        assert result[0] < 2.0  # bounded, not exploding


# --- Equation 10b: Attention Entropy Gate Signal ---


class TestAttentionEntropyGate:
    """H_out = -Σ(p_i · log(p_i)) / log(n) — penalizes broad retrieval patterns."""

    def test_uniform_attention_gives_high_entropy(self):
        """When output is equally similar to all keys → H ≈ 1.0."""
        config = MemoryConfig(embedding_dim=64, T_entropy=0.5)
        layer = make_layer(config)

        # Create 5 memories with orthogonal keys
        for i in range(5):
            k = np.zeros(64, dtype=np.float32)
            k[i * 10] = 1.0  # orthogonal basis vectors
            v = random_embedding(64, np.random.default_rng(100 + i))
            layer._create_memory(k, v, f"fact {i}", f"ok {i}")

        # Output that's equally similar to all keys (or close to it)
        uniform_out = np.ones(64, dtype=np.float32)
        uniform_out = uniform_out / np.linalg.norm(uniform_out)

        H = layer._compute_output_entropy(uniform_out)
        # With uniform-ish attention, entropy should be high
        assert H > 0.5

    def test_focused_attention_gives_low_entropy(self):
        """When output is very similar to one key and dissimilar to others → H ≈ 0."""
        config = MemoryConfig(
            embedding_dim=64, T_entropy=0.1
        )  # low T sharpens distribution
        layer = make_layer(config)

        # Create memories with well-separated keys
        for i in range(5):
            k = np.zeros(64, dtype=np.float32)
            k[i * 10] = 1.0
            v = random_embedding(64, np.random.default_rng(100 + i))
            layer._create_memory(k, v, f"fact {i}", f"ok {i}")

        # Output focused on key 0
        focused_out = np.zeros(64, dtype=np.float32)
        focused_out[0] = 1.0

        H = layer._compute_output_entropy(focused_out)
        # Focused attention → low entropy
        assert H < 0.5

    def test_entropy_empty_memories_returns_zero(self):
        """With no memories, entropy should be 0."""
        config = MemoryConfig(embedding_dim=64)
        layer = make_layer(config)
        output = random_embedding(64, np.random.default_rng(1))
        assert layer._compute_output_entropy(output) == 0.0

    def test_entropy_single_memory_returns_zero(self):
        """With 1 memory, entropy should be 0 (need >=2 for distribution)."""
        config = MemoryConfig(embedding_dim=64)
        layer = make_layer(config)
        k = random_embedding(64, np.random.default_rng(1))
        v = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(k, v, "test", "ok")
        output = random_embedding(64, np.random.default_rng(3))
        assert layer._compute_output_entropy(output) == 0.0

    def test_entropy_penalizes_gate(self):
        """High entropy should push the gate toward closing (lower gate value)."""
        config = MemoryConfig(
            embedding_dim=64,
            beta_gate=6.0,
            theta_gate=0.4,
            gate_w_entropy=0.15,
            gate_w_out=0.0,
            gate_w_val=0.0,
            gate_w_ctx=0.0,
        )
        layer = make_layer(config)

        # Gate with zero entropy
        gate_no_entropy = layer._memory_input_gate(
            R_overlap=0.3, best_sim=0.3, H_out=0.0
        )
        # Gate with high entropy
        gate_high_entropy = layer._memory_input_gate(
            R_overlap=0.3, best_sim=0.3, H_out=0.9
        )

        assert gate_high_entropy < gate_no_entropy

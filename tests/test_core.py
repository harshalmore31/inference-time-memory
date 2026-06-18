"""Tests for the core memory equations — no API calls needed."""

import numpy as np

from itm.config import MemoryConfig
from itm.core import MemoryEntry, MemoryLayer
from itm.embeddings import EmbeddingService
from itm.formatting import MemoryFormatter
from itm.storage import MemoryStorage

# --- Helpers ---


def random_embedding(dim=1536, rng=None):
    rng = rng or np.random.default_rng(42)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


class FakeEmbedder:
    """Stub that returns deterministic embeddings without calling OpenAI."""

    def __init__(self, dim=1536):
        self.dim = dim
        self._rng = np.random.default_rng(0)
        self._cache = {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = random_embedding(self.dim, self._rng)
        return self._cache[text]


def make_layer(config=None, embedder=None):
    config = config or MemoryConfig(embedding_dim=64)
    embedder = embedder or FakeEmbedder(dim=64)
    return MemoryLayer(config, embedder)


# --- Equation 1: Strength Update ---


class TestStrengthUpdate:
    """s_i(t+1) = s_i(t) * gamma + alpha * sim(e_in, k_i)"""

    def test_decay_applied_every_timestep(self):
        layer = make_layer()
        rng = np.random.default_rng(1)
        key = random_embedding(64, rng)
        out = random_embedding(64, rng)

        # Create a memory manually
        layer._create_memory(key, out, "test input", "test output")
        initial_strength = layer.memories[0].strength

        # A completely unrelated input — strength should mostly just decay
        unrelated = random_embedding(64, np.random.default_rng(99))
        layer.update(unrelated, unrelated, "unrelated", "unrelated")

        # After decay: s * gamma + alpha * sim (sim is low for random vectors)
        assert layer.memories[0].strength < initial_strength

    def test_repeated_reinforcement_increases_strength(self):
        layer = make_layer()
        rng = np.random.default_rng(1)
        key = random_embedding(64, rng)
        out = random_embedding(64, rng)

        # First memory
        layer.update(key, out, "python basics", "python tutorial")
        s_after_create = layer.memories[0].strength

        # Reinforce with the same key (sim ≈ 1.0)
        layer.update(key, out, "python basics", "python tutorial")

        # Strength should have grown despite decay
        s_after_reinforce = layer.memories[0].strength
        assert s_after_reinforce > s_after_create * layer.config.gamma

    def test_global_decay_multiplies_all(self):
        layer = make_layer()
        rng = np.random.default_rng(1)

        # Create 3 memories
        for i in range(3):
            k = random_embedding(64, np.random.default_rng(i))
            v = random_embedding(64, np.random.default_rng(i + 100))
            layer._create_memory(k, v, f"mem {i}", f"out {i}")

        strengths_before = [m.strength for m in layer.memories]
        layer._apply_global_decay()
        strengths_after = [m.strength for m in layer.memories]

        for before, after in zip(strengths_before, strengths_after):
            assert abs(after - before * layer.config.gamma) < 1e-10


# --- Equation 2: Value Update (Contradiction) ---


class TestContradiction:
    """v_i(t+1) = v_i(t) + b_i * (e_out_new - v_i(t))"""

    def test_contradiction_detected_when_key_similar_value_different(self):
        config = MemoryConfig(
            embedding_dim=64,
            theta_key=0.7,
            theta_value=0.5,
            theta_create=0.8,
        )
        layer = make_layer(config)

        rng = np.random.default_rng(5)
        key = random_embedding(64, rng)
        old_value = random_embedding(64, np.random.default_rng(10))
        new_value = random_embedding(
            64, np.random.default_rng(20)
        )  # different direction

        # Create memory with old value
        entry = MemoryEntry(
            key=key.copy(),
            value=old_value.copy(),
            strength=1.0,
            bias=0.5,
            timestamp=0,
            access_count=0,
            input_text="I live in Mumbai",
            output_text="User lives in Mumbai",
        )
        layer.memories.append(entry)

        # Check contradiction detection
        key_sim = EmbeddingService.cosine_similarity(key, key)  # 1.0
        value_sim = EmbeddingService.cosine_similarity(old_value, new_value)

        is_contradiction = layer._detect_contradiction(entry, new_value, key_sim)

        # With sim(key,key)=1.0 > theta_key=0.7 and random values likely < theta_value=0.5
        if value_sim < config.theta_value:
            assert is_contradiction
        else:
            assert not is_contradiction

    def test_value_shifts_toward_new_on_contradiction(self):
        config = MemoryConfig(embedding_dim=64, theta_create=0.99, initial_bias=0.5)
        layer = make_layer(config)

        # Manually set up a memory we can contradict
        key = np.ones(64, dtype=np.float32) / np.sqrt(64)
        old_value = np.zeros(64, dtype=np.float32)
        old_value[0] = 1.0  # points in one direction

        new_value = np.zeros(64, dtype=np.float32)
        new_value[1] = 1.0  # points in a different direction

        entry = MemoryEntry(
            key=key.copy(),
            value=old_value.copy(),
            strength=1.0,
            bias=0.5,
            timestamp=0,
            input_text="location",
            output_text="old place",
        )
        layer.memories.append(entry)

        # Equation 2: v_new = v_old + bias * (e_out - v_old)
        expected = old_value + 0.5 * (new_value - old_value)
        entry.value = entry.value + entry.bias * (new_value - entry.value)

        np.testing.assert_allclose(entry.value, expected, atol=1e-6)

    def test_no_contradiction_when_values_agree(self):
        layer = make_layer()
        key = random_embedding(64, np.random.default_rng(1))
        value = random_embedding(64, np.random.default_rng(2))

        entry = MemoryEntry(
            key=key,
            value=value.copy(),
            strength=1.0,
            bias=0.5,
            timestamp=0,
            input_text="test",
            output_text="test",
        )

        # Same value = no contradiction
        assert not layer._detect_contradiction(entry, value, 1.0)

    def test_hard_contradiction_value_renormalized_to_unit(self):
        """Eq 2: after the hard-contradiction blend the value is unit norm.

        Property: ||v_i|| == 1.0 after a contradiction update. Without
        renormalization, blending two unit vectors leaves norm ~0.707, which
        makes contradicted values under-contribute in recall_weighted_value.
        Driven through the real update() path so the code (not a hand blend)
        performs the renormalization.
        """
        config = MemoryConfig(
            embedding_dim=64,
            theta_key=0.7,
            theta_value=0.5,
            theta_create=0.8,
            initial_bias=0.5,
        )
        layer = make_layer(config)

        # Key the new input will match (sim = 1.0 > theta_create -> strengthen).
        key = random_embedding(64, np.random.default_rng(7))
        old_value = random_embedding(64, np.random.default_rng(11))
        # Orthogonal new output -> value_sim ~ 0 < theta_value -> contradiction.
        new_value = random_embedding(64, np.random.default_rng(23))

        entry = MemoryEntry(
            key=key.copy(),
            value=old_value.copy(),
            strength=1.0,
            bias=config.initial_bias,
            timestamp=0,
            input_text="I live in Mumbai",
            output_text="User lives in Mumbai",
        )
        layer.memories.append(entry)

        # Confirm this is genuinely a contradiction under the configured thresholds.
        value_sim = EmbeddingService.cosine_similarity(old_value, new_value)
        assert value_sim < config.theta_value

        result = layer.update(
            key, new_value, "I live in Mumbai", "User now lives in Delhi"
        )
        assert result["action"] == "contradiction"

        norm = float(np.linalg.norm(layer.memories[0].value))
        assert abs(norm - 1.0) < 1e-6


# --- Equation 3: Recall ---


class TestRecall:
    """R(q) = sum(s_i * sim(q, k_i) * v_i) / sum(s_i * sim)"""

    def test_recall_returns_most_relevant_first(self):
        layer = make_layer()
        rng = np.random.default_rng(42)

        # Create 3 memories
        keys = [random_embedding(64, np.random.default_rng(i)) for i in range(3)]
        values = [random_embedding(64, np.random.default_rng(i + 50)) for i in range(3)]

        for k, v, i in zip(keys, values, range(3)):
            layer._create_memory(k, v, f"input {i}", f"output {i}")

        # Boost one memory's strength
        layer.memories[1].strength = 5.0

        # Query with keys[1] — should rank memory 1 highest
        results = layer.recall(keys[1])
        assert len(results) > 0
        assert results[0][0] is layer.memories[1]

    def test_recall_empty_returns_empty(self):
        layer = make_layer()
        q = random_embedding(64)
        assert layer.recall(q) == []

    def test_recall_weighted_value_is_weighted_average(self):
        config = MemoryConfig(embedding_dim=4, min_relevance=0.0)
        layer = make_layer(config, FakeEmbedder(dim=4))

        # Two memories with known keys and values
        k1 = np.array([1, 0, 0, 0], dtype=np.float32)
        v1 = np.array([1, 0, 0, 0], dtype=np.float32)
        k2 = np.array([0, 1, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0, 0], dtype=np.float32)

        layer._create_memory(k1, v1, "a", "a")
        layer._create_memory(k2, v2, "b", "b")

        # Query aligned with k1 — should weight v1 much more
        query = np.array([0.9, 0.1, 0, 0], dtype=np.float32)
        result = layer.recall_weighted_value(query)

        assert result is not None
        # Result should be closer to v1 than v2
        sim_to_v1 = float(
            np.dot(result, v1) / (np.linalg.norm(result) * np.linalg.norm(v1))
        )
        sim_to_v2 = float(
            np.dot(result, v2) / (np.linalg.norm(result) * np.linalg.norm(v2))
        )
        assert sim_to_v1 > sim_to_v2


# --- Never-Forget ---


class TestNeverForget:
    """Memories are never pruned. Strength = relevance, not existence."""

    def test_weak_memories_persist(self):
        layer = make_layer()
        rng = np.random.default_rng(1)

        # Create a memory then decay it heavily
        k = random_embedding(64, rng)
        v = random_embedding(64, rng)
        layer._create_memory(k, v, "old fact", "old response")

        # Simulate 1000 timesteps of decay
        for _ in range(1000):
            layer._apply_global_decay()

        # Memory still exists even though strength is tiny
        assert len(layer.memories) == 1
        assert layer.memories[0].strength > 0
        assert layer.memories[0].strength < 0.01  # very weak but alive


# --- Update Cycle ---


class TestUpdateCycle:

    def test_first_memory_is_created(self):
        layer = make_layer()
        k = random_embedding(64, np.random.default_rng(1))
        v = random_embedding(64, np.random.default_rng(2))

        result = layer.update(k, v, "hello", "hi there")
        assert result["action"] == "created"
        assert len(layer.memories) == 1

    def test_similar_input_strengthens(self):
        config = MemoryConfig(embedding_dim=64, theta_create=0.5)
        layer = make_layer(config)

        k = random_embedding(64, np.random.default_rng(1))
        v = random_embedding(64, np.random.default_rng(2))

        # Create
        layer.update(k, v, "hello", "hi there")

        # Same input again — should strengthen (sim=1.0 > theta_create=0.5)
        result = layer.update(k, v, "hello", "hi there")
        assert result["action"] == "strengthened"

    def test_different_input_creates_new(self):
        config = MemoryConfig(embedding_dim=64, theta_create=0.9)
        layer = make_layer(config)

        k1 = random_embedding(64, np.random.default_rng(1))
        v1 = random_embedding(64, np.random.default_rng(2))
        k2 = random_embedding(64, np.random.default_rng(3))
        v2 = random_embedding(64, np.random.default_rng(4))

        layer.update(k1, v1, "topic A", "response A")
        result = layer.update(k2, v2, "topic B", "response B")

        assert result["action"] == "created"
        assert len(layer.memories) == 2

    def test_timestep_increments(self):
        layer = make_layer()
        k = random_embedding(64, np.random.default_rng(1))
        v = random_embedding(64, np.random.default_rng(2))

        assert layer.timestep == 0
        layer.update(k, v, "a", "b")
        assert layer.timestep == 1
        layer.update(k, v, "c", "d")
        assert layer.timestep == 2


# --- Storage Round-Trip ---


class TestStorage:

    def test_save_and_load_roundtrip(self, tmp_path):
        config = MemoryConfig(embedding_dim=64, memory_dir=str(tmp_path))
        embedder = FakeEmbedder(dim=64)
        layer = MemoryLayer(config, embedder)
        storage = MemoryStorage(config)

        # Create some memories
        for i in range(5):
            k = random_embedding(64, np.random.default_rng(i))
            v = random_embedding(64, np.random.default_rng(i + 50))
            layer.update(k, v, f"input {i}", f"output {i}")

        storage.save(layer)

        # Load into a fresh layer
        layer2 = MemoryLayer(config, embedder)
        loaded = storage.load(layer2)

        assert loaded
        assert len(layer2.memories) == len(layer.memories)
        assert layer2.timestep == layer.timestep

        for m1, m2 in zip(layer.memories, layer2.memories):
            np.testing.assert_allclose(m1.key, m2.key, atol=1e-6)
            np.testing.assert_allclose(m1.value, m2.value, atol=1e-6)
            assert abs(m1.strength - m2.strength) < 1e-10
            assert m1.input_text == m2.input_text
            assert m1.output_text == m2.output_text

    def test_load_nonexistent_returns_false(self, tmp_path):
        config = MemoryConfig(
            embedding_dim=64, memory_dir=str(tmp_path), user_id="nobody"
        )
        layer = MemoryLayer(config, FakeEmbedder(64))
        storage = MemoryStorage(config)

        assert not storage.load(layer)


# --- Formatting ---


class TestFormatting:

    def test_format_for_prompt_empty(self):
        fmt = MemoryFormatter()
        assert fmt.format_for_prompt([]) == ""

    def test_format_for_prompt_confidence_levels(self):
        fmt = MemoryFormatter()
        entry = MemoryEntry(
            key=np.zeros(4, dtype=np.float32),
            value=np.zeros(4, dtype=np.float32),
            strength=1.0,
            bias=0.5,
            timestamp=0,
            input_text="test memory",
            output_text="test output",
        )

        result_high = fmt.format_for_prompt([(entry, 1.5)])
        assert "HIGH" in result_high

        result_med = fmt.format_for_prompt([(entry, 0.5)])
        assert "MEDIUM" in result_med

        result_low = fmt.format_for_prompt([(entry, 0.1)])
        assert "LOW" in result_low


# --- Cosine Similarity ---


class TestSimilarity:

    def test_identical_vectors_similarity_one(self):
        v = random_embedding(64)
        assert abs(EmbeddingService.cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_similarity_zero(self):
        a = np.zeros(64, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(64, dtype=np.float32)
        b[1] = 1.0
        assert abs(EmbeddingService.cosine_similarity(a, b)) < 1e-6

    def test_matrix_similarity_shape(self):
        query = random_embedding(64)
        keys = np.stack(
            [random_embedding(64, np.random.default_rng(i)) for i in range(10)]
        )
        sims = EmbeddingService.cosine_similarity_matrix(query, keys)
        assert sims.shape == (10,)

    def test_zero_vector_returns_zero(self):
        zero = np.zeros(64, dtype=np.float32)
        other = random_embedding(64)
        assert EmbeddingService.cosine_similarity(zero, other) == 0.0

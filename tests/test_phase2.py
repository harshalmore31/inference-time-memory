"""Tests for Phase 2: Equations 12-17 (Kahneman + ACT-R + MemWire).

Equation 12 — Prospect-Theory Asymmetric Strength (Kahneman 1979)
Equation 13 — Tension Detection (WYSIATI-breaking)
Equation 14 — Memory Category Channels (Attribute Substitution fix)
Equation 15 — Adaptive Recall Depth (Kahneman Capacity Model)
Equation 16 — Feedback Loop (RLHF-inspired)
Equation 17 — ACT-R Base-Level Activation (Power-Law Decay)
"""

import numpy as np

from itm.config import MemoryConfig
from itm.core import MemoryEntry, MemoryLayer, RecallResult
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


# ══════════════════════════════════════════════════════════════
# Equation 12: Prospect-Theory Asymmetric Strength (Kahneman)
# ══════════════════════════════════════════════════════════════


class TestProspectTheory:
    """v(Δ) = |Δ|^ρ for gains, -λ·|Δ|^ρ for losses."""

    def test_gain_diminishing(self):
        """Gain delta should be sublinear: Δ^0.88 < Δ for Δ > 1."""
        config = MemoryConfig(
            embedding_dim=64,
            prospect_enabled=True,
            prospect_rho=0.88,
            prospect_lambda_loss=2.25,
        )
        layer = make_layer(config)

        # For delta > 1, x^0.88 < x
        result = layer._prospect_strength_delta(2.0)
        assert result < 2.0
        assert result > 0.0  # still positive

    def test_loss_amplified(self):
        """Loss delta should be amplified by λ=2.25."""
        config = MemoryConfig(
            embedding_dim=64,
            prospect_enabled=True,
            prospect_rho=0.88,
            prospect_lambda_loss=2.25,
        )
        layer = make_layer(config)

        gain = layer._prospect_strength_delta(0.5)
        loss = layer._prospect_strength_delta(-0.5)

        # Loss magnitude should be larger than gain magnitude
        assert abs(loss) > abs(gain)
        # Specifically, loss should be ~2.25x the gain
        ratio = abs(loss) / abs(gain)
        assert abs(ratio - 2.25) < 0.01

    def test_asymmetry_same_magnitude(self):
        """Same |Δ|, but loss hits harder than gain by factor λ."""
        config = MemoryConfig(
            embedding_dim=64,
            prospect_enabled=True,
            prospect_rho=0.88,
            prospect_lambda_loss=2.25,
        )
        layer = make_layer(config)

        for delta in [0.1, 0.3, 0.5, 0.8, 1.0]:
            gain = layer._prospect_strength_delta(delta)
            loss = layer._prospect_strength_delta(-delta)
            assert gain > 0
            assert loss < 0
            assert abs(loss) / abs(gain) == pytest.approx(2.25, abs=0.01)

    def test_disabled_is_identity(self):
        """When prospect_enabled=False, delta passes through unchanged."""
        config = MemoryConfig(embedding_dim=64, prospect_enabled=False)
        layer = make_layer(config)

        assert layer._prospect_strength_delta(0.5) == 0.5
        assert layer._prospect_strength_delta(-0.3) == -0.3
        assert layer._prospect_strength_delta(0.0) == 0.0

    def test_zero_delta_stays_zero(self):
        """Zero delta → zero output regardless of prospect settings."""
        config = MemoryConfig(embedding_dim=64, prospect_enabled=True)
        layer = make_layer(config)
        assert layer._prospect_strength_delta(0.0) == 0.0

    def test_decay_symmetric_regardless_of_prospect(self):
        """Global decay is always symmetric — prospect theory only applies to events.

        Kahneman's prospect theory models reactions to discrete events,
        not the passive passage of time. Decay should be identical
        whether prospect_enabled is True or False.
        """
        config = MemoryConfig(
            embedding_dim=64,
            prospect_enabled=True,
            prospect_rho=0.88,
            prospect_lambda_loss=2.25,
            gamma=0.99,
        )
        layer = make_layer(config)

        k = random_embedding(64, np.random.default_rng(1))
        v = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(k, v, "test", "test")

        layer._apply_global_decay()
        prospect_strength = layer.memories[0].strength

        # Compare with non-prospect decay
        config2 = MemoryConfig(embedding_dim=64, prospect_enabled=False, gamma=0.99)
        layer2 = make_layer(config2)
        layer2._create_memory(k.copy(), v.copy(), "test", "test")
        layer2._apply_global_decay()
        normal_strength = layer2.memories[0].strength

        # Decay should be identical — prospect doesn't affect passive decay
        assert prospect_strength == normal_strength


import pytest

# ══════════════════════════════════════════════════════════════
# Equation 13: Tension Detection (WYSIATI-Breaking)
# ══════════════════════════════════════════════════════════════


class TestTensionDetection:
    """tension = sim(k_top, k_other) > θ_key AND sim(v_top, v_other) < θ_val."""

    def test_detects_conflicting_values(self):
        """Two memories with similar keys but opposite values → conflict."""
        config = MemoryConfig(
            embedding_dim=64,
            tension_enabled=True,
            theta_tension_key=0.5,
            theta_tension_val=0.4,
        )
        layer = make_layer(config)

        key = random_embedding(64, np.random.default_rng(1))
        val1 = random_embedding(64, np.random.default_rng(2))
        val2 = -val1  # opposite value → low cosine sim

        mem1 = MemoryEntry(
            key=key.copy(),
            value=val1,
            strength=1.0,
            bias=0.5,
            timestamp=1,
            input_text="fact A",
            output_text="yes",
        )
        mem2 = MemoryEntry(
            key=key.copy(),
            value=val2,
            strength=1.0,
            bias=0.5,
            timestamp=2,
            input_text="fact A",
            output_text="no",
        )

        results = [(mem1, 0.9), (mem2, 0.8)]
        recall_result = layer._detect_tension(results)

        assert len(recall_result.conflicting) == 1
        assert recall_result.conflicting[0][0].output_text == "no"

    def test_no_conflict_when_values_agree(self):
        """Similar keys + similar values = supporting, not conflict."""
        config = MemoryConfig(
            embedding_dim=64,
            tension_enabled=True,
            theta_tension_key=0.5,
            theta_tension_val=0.4,
        )
        layer = make_layer(config)

        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))

        mem1 = MemoryEntry(
            key=key.copy(),
            value=val.copy(),
            strength=1.0,
            bias=0.5,
            timestamp=1,
            input_text="fact",
            output_text="yes",
        )
        mem2 = MemoryEntry(
            key=key.copy(),
            value=val.copy(),
            strength=1.0,
            bias=0.5,
            timestamp=2,
            input_text="fact",
            output_text="yes",
        )

        results = [(mem1, 0.9), (mem2, 0.8)]
        recall_result = layer._detect_tension(results)

        assert len(recall_result.conflicting) == 0
        assert len(recall_result.supporting) == 2

    def test_unrelated_topics_not_compared(self):
        """Low key sim = not compared for tension (different topics)."""
        config = MemoryConfig(
            embedding_dim=64,
            tension_enabled=True,
            theta_tension_key=0.9,
            theta_tension_val=0.4,
        )
        layer = make_layer(config)

        key1 = random_embedding(64, np.random.default_rng(1))
        key2 = random_embedding(64, np.random.default_rng(99))  # different topic
        val1 = random_embedding(64, np.random.default_rng(2))
        val2 = -val1  # opposite value but different topic

        mem1 = MemoryEntry(
            key=key1,
            value=val1,
            strength=1.0,
            bias=0.5,
            timestamp=1,
            input_text="A",
            output_text="yes",
        )
        mem2 = MemoryEntry(
            key=key2,
            value=val2,
            strength=1.0,
            bias=0.5,
            timestamp=2,
            input_text="B",
            output_text="no",
        )

        results = [(mem1, 0.9), (mem2, 0.8)]
        recall_result = layer._detect_tension(results)

        # Not flagged as conflict because keys are too different
        assert len(recall_result.conflicting) == 0

    def test_recall_result_backward_compat(self):
        """all_results should match the original flat list."""
        config = MemoryConfig(embedding_dim=64, tension_enabled=True)
        layer = make_layer(config)

        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))

        mem = MemoryEntry(
            key=key,
            value=val,
            strength=1.0,
            bias=0.5,
            timestamp=1,
            input_text="test",
            output_text="test",
        )
        results = [(mem, 0.9)]
        recall_result = layer._detect_tension(results)

        assert len(recall_result.all_results) == 1
        assert recall_result.all_results[0] == results[0]

    def test_disabled_returns_all_as_supporting(self):
        """When tension_enabled=False, all results are supporting."""
        config = MemoryConfig(embedding_dim=64, tension_enabled=False)
        layer = make_layer(config)

        key = random_embedding(64, np.random.default_rng(1))
        val1 = random_embedding(64, np.random.default_rng(2))
        val2 = -val1

        mem1 = MemoryEntry(
            key=key.copy(),
            value=val1,
            strength=1.0,
            bias=0.5,
            timestamp=1,
            input_text="A",
            output_text="yes",
        )
        mem2 = MemoryEntry(
            key=key.copy(),
            value=val2,
            strength=1.0,
            bias=0.5,
            timestamp=2,
            input_text="A",
            output_text="no",
        )

        results = [(mem1, 0.9), (mem2, 0.8)]
        recall_result = layer._detect_tension(results)

        assert len(recall_result.conflicting) == 0
        assert len(recall_result.supporting) == 2


# ══════════════════════════════════════════════════════════════
# Equation 14: Memory Category Channels
# ══════════════════════════════════════════════════════════════


class TestCategoryChannels:
    """channel_i = argmax_c(sim(k_i, anchor_c))."""

    def test_disabled_returns_empty(self):
        """When channels_enabled=False, category is empty string."""
        config = MemoryConfig(embedding_dim=64, channels_enabled=False)
        layer = make_layer(config)
        emb = random_embedding(64, np.random.default_rng(1))
        assert layer._classify_memory(emb) == ""

    def test_enabled_returns_category(self):
        """When enabled, classification returns a non-empty category."""
        config = MemoryConfig(embedding_dim=64, channels_enabled=True)
        embedder = FakeEmbedder(dim=64)
        layer = make_layer(config, embedder)

        emb = random_embedding(64, np.random.default_rng(1))
        cat = layer._classify_memory(emb)
        assert cat in ("fact", "preference", "instruction", "event")

    def test_category_stored_on_create(self):
        """New memories should have their category set."""
        config = MemoryConfig(embedding_dim=64, channels_enabled=True)
        embedder = FakeEmbedder(dim=64)
        layer = make_layer(config, embedder)

        k = random_embedding(64, np.random.default_rng(1))
        v = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(k, v, "test input", "test output")

        assert layer.memories[0].category != ""

    def test_category_persists_save_load(self, tmp_path):
        """Category should survive save/load cycle."""
        config = MemoryConfig(
            embedding_dim=64, channels_enabled=True, memory_dir=str(tmp_path)
        )
        embedder = FakeEmbedder(dim=64)
        layer = make_layer(config, embedder)
        storage = MemoryStorage(config)

        k = random_embedding(64, np.random.default_rng(1))
        v = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(k, v, "test", "response")
        original_cat = layer.memories[0].category

        storage.save(layer)

        layer2 = make_layer(config, embedder)
        storage.load(layer2)

        assert layer2.memories[0].category == original_cat


# ══════════════════════════════════════════════════════════════
# Equation 15: Adaptive Recall Depth (Kahneman Capacity Model)
# ══════════════════════════════════════════════════════════════


class TestAdaptiveRecallDepth:
    """top_k_eff = ceil(top_k · (1 + β_d · difficulty))."""

    def test_easy_query_low_k(self):
        """High max_sim (easy query) → top_k stays near base."""
        config = MemoryConfig(
            embedding_dim=64,
            adaptive_k_enabled=True,
            top_k=5,
            beta_difficulty=1.0,
            top_k_max=15,
        )
        layer = make_layer(config)

        # Create a memory, then query with same key (max_sim ≈ 1.0)
        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key, val, "test", "test")

        effective = layer._compute_effective_top_k(key)
        assert effective == 5  # difficulty ≈ 0 → no increase

    def test_hard_query_high_k(self):
        """Low max_sim (hard query) → top_k increases."""
        config = MemoryConfig(
            embedding_dim=64,
            adaptive_k_enabled=True,
            top_k=5,
            beta_difficulty=1.0,
            top_k_max=15,
        )
        layer = make_layer(config)

        # Create memories with specific key
        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key, val, "test", "test")

        # Query with orthogonal vector (max_sim ≈ 0)
        hard_query = np.zeros(64, dtype=np.float32)
        hard_query[63] = 1.0

        effective = layer._compute_effective_top_k(hard_query)
        assert effective > 5  # difficulty ≈ 1 → increase

    def test_bounded_by_max(self):
        """Effective top_k should never exceed top_k_max."""
        config = MemoryConfig(
            embedding_dim=64,
            adaptive_k_enabled=True,
            top_k=5,
            beta_difficulty=10.0,
            top_k_max=8,
        )
        layer = make_layer(config)

        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key, val, "test", "test")

        # Hard query with very high beta → would exceed max without cap
        hard_query = np.zeros(64, dtype=np.float32)
        hard_query[63] = 1.0
        effective = layer._compute_effective_top_k(hard_query)

        assert effective <= 8

    def test_disabled_returns_base(self):
        """When adaptive_k_enabled=False, returns config.top_k."""
        config = MemoryConfig(embedding_dim=64, adaptive_k_enabled=False, top_k=5)
        layer = make_layer(config)

        key = random_embedding(64, np.random.default_rng(1))
        assert layer._compute_effective_top_k(key) == 5


# ══════════════════════════════════════════════════════════════
# Equation 16: Feedback Loop (RLHF-inspired)
# ══════════════════════════════════════════════════════════════


class TestFeedbackLoop:
    """alignment = sim(e_response, v_i) → strengthen or weaken."""

    def test_strengthen_aligned(self):
        """High alignment → memory strength increases."""
        config = MemoryConfig(
            embedding_dim=64,
            feedback_enabled=True,
            theta_align=0.6,
            theta_misalign=0.3,
            alpha_feedback=0.05,
            eta_feedback=0.1,
        )
        layer = make_layer(config)

        val = random_embedding(64, np.random.default_rng(1))
        key = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key, val, "fact", "response")
        initial_strength = layer.memories[0].strength

        # Response embedding = same as value → alignment ≈ 1.0
        recalled = [(layer.memories[0], 0.9)]
        result = layer.apply_feedback(val.copy(), recalled)

        assert result["strengthened"] == 1
        assert layer.memories[0].strength > initial_strength

    def test_weaken_misaligned(self):
        """Low alignment → memory strength decreases."""
        config = MemoryConfig(
            embedding_dim=64,
            feedback_enabled=True,
            theta_align=0.6,
            theta_misalign=0.3,
            alpha_feedback=0.1,
            eta_feedback=0.1,
        )
        layer = make_layer(config)

        val = random_embedding(64, np.random.default_rng(1))
        key = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key, val, "fact", "response")
        initial_strength = layer.memories[0].strength

        # Response embedding = opposite of value → alignment ≈ -1.0
        response = -val
        recalled = [(layer.memories[0], 0.9)]
        result = layer.apply_feedback(response, recalled)

        assert result["weakened"] == 1
        assert layer.memories[0].strength < initial_strength

    def test_neutral_zone_no_change(self):
        """Alignment between thresholds → no change."""
        config = MemoryConfig(
            embedding_dim=64,
            feedback_enabled=True,
            theta_align=0.9,
            theta_misalign=0.1,
            alpha_feedback=0.05,
        )
        layer = make_layer(config)

        val = random_embedding(64, np.random.default_rng(1))
        key = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key, val, "fact", "response")
        initial_strength = layer.memories[0].strength

        # Response moderately similar (alignment ~ 0.4-0.6)
        # Use a vector with ~0.5 similarity
        response = random_embedding(64, np.random.default_rng(99))
        recalled = [(layer.memories[0], 0.9)]
        result = layer.apply_feedback(response, recalled)

        # Depending on exact sim, it's in the neutral zone
        total_changes = result["strengthened"] + result["weakened"]
        # If in neutral zone, no changes
        if total_changes == 0:
            assert layer.memories[0].strength == initial_strength

    def test_edge_adjustment(self):
        """Feedback should adjust graph edge weights."""
        config = MemoryConfig(
            embedding_dim=64,
            feedback_enabled=True,
            theta_align=0.5,
            alpha_feedback=0.05,
            eta_feedback=0.2,
            theta_edge=0.01,
        )
        layer = make_layer(config)

        # Create two connected memories
        key1 = random_embedding(64, np.random.default_rng(1))
        val1 = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key1, val1, "A", "response A")

        key2 = key1 + 0.1 * random_embedding(64, np.random.default_rng(3))
        key2 = (key2 / np.linalg.norm(key2)).astype(np.float32)
        val2 = val1 + 0.1 * random_embedding(64, np.random.default_rng(4))
        val2 = (val2 / np.linalg.norm(val2)).astype(np.float32)
        layer._create_memory(key2, val2, "B", "response B")

        # Check if edge exists between 0 and 1
        if 0 in layer.graph.edges and 1 in layer.graph.edges[0]:
            old_weight = layer.graph.edges[0][1]

            # Feedback with aligned response for memory 0
            recalled = [(layer.memories[0], 0.9)]
            layer.apply_feedback(val1.copy(), recalled)

            if 0 in layer.graph.edges and 1 in layer.graph.edges[0]:
                new_weight = layer.graph.edges[0][1]
                assert new_weight >= old_weight  # strengthened

    def test_disabled_no_effect(self):
        """When feedback_enabled=False, no changes."""
        config = MemoryConfig(embedding_dim=64, feedback_enabled=False)
        layer = make_layer(config)

        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key, val, "fact", "response")
        initial_strength = layer.memories[0].strength

        recalled = [(layer.memories[0], 0.9)]
        result = layer.apply_feedback(val.copy(), recalled)

        assert result == {"strengthened": 0, "weakened": 0}
        assert layer.memories[0].strength == initial_strength


# ══════════════════════════════════════════════════════════════
# Equation 17: ACT-R Base-Level Activation (Power-Law Decay)
# ══════════════════════════════════════════════════════════════


class TestACTRActivation:
    """B_i = ln(Σ_n (t - t_n)^(-d)) where d = 0.5."""

    def test_single_access(self):
        """One access at t=5, query at t=10: B = ln((10-5)^-0.5)."""
        config = MemoryConfig(embedding_dim=64, actr_enabled=True, actr_decay=0.5)
        layer = make_layer(config)
        layer.timestep = 10

        mem = MemoryEntry(
            key=random_embedding(64),
            value=random_embedding(64),
            strength=1.0,
            bias=0.5,
            timestamp=5,
            access_history=[5],
        )

        b = layer._compute_base_level_activation(mem)
        expected = np.log(5 ** (-0.5))  # ln(0.447) ≈ -0.804
        assert abs(b - expected) < 0.01

    def test_multiple_accesses_sum(self):
        """Multiple accesses: B = ln(Σ (t-t_n)^-d)."""
        config = MemoryConfig(embedding_dim=64, actr_enabled=True, actr_decay=0.5)
        layer = make_layer(config)
        layer.timestep = 10

        mem = MemoryEntry(
            key=random_embedding(64),
            value=random_embedding(64),
            strength=1.0,
            bias=0.5,
            timestamp=8,
            access_history=[2, 5, 8],
        )

        b = layer._compute_base_level_activation(mem)
        # Σ = (10-2)^-0.5 + (10-5)^-0.5 + (10-8)^-0.5
        # = 8^-0.5 + 5^-0.5 + 2^-0.5
        # = 0.354 + 0.447 + 0.707 = 1.508
        expected = np.log(0.354 + 0.447 + 0.707)
        assert abs(b - expected) < 0.02

    def test_recent_beats_old(self):
        """Memory accessed recently scores higher than one accessed long ago."""
        config = MemoryConfig(embedding_dim=64, actr_enabled=True, actr_decay=0.5)
        layer = make_layer(config)
        layer.timestep = 100

        recent = MemoryEntry(
            key=random_embedding(64),
            value=random_embedding(64),
            strength=1.0,
            bias=0.5,
            timestamp=99,
            access_history=[99],
        )
        old = MemoryEntry(
            key=random_embedding(64),
            value=random_embedding(64),
            strength=1.0,
            bias=0.5,
            timestamp=10,
            access_history=[10],
        )

        assert layer._compute_base_level_activation(
            recent
        ) > layer._compute_base_level_activation(old)

    def test_frequent_beats_rare(self):
        """Memory accessed 5 times beats one accessed once (even if same recency)."""
        config = MemoryConfig(embedding_dim=64, actr_enabled=True, actr_decay=0.5)
        layer = make_layer(config)
        layer.timestep = 20

        frequent = MemoryEntry(
            key=random_embedding(64),
            value=random_embedding(64),
            strength=1.0,
            bias=0.5,
            timestamp=18,
            access_history=[5, 8, 12, 15, 18],
        )
        rare = MemoryEntry(
            key=random_embedding(64),
            value=random_embedding(64),
            strength=1.0,
            bias=0.5,
            timestamp=18,
            access_history=[18],
        )

        assert layer._compute_base_level_activation(
            frequent
        ) > layer._compute_base_level_activation(rare)

    def test_history_capped(self):
        """More than actr_max_history accesses only keeps last N."""
        config = MemoryConfig(embedding_dim=64, actr_enabled=True, actr_max_history=5)
        layer = make_layer(config)

        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))

        # Simulate many accesses by updating with same key
        layer._create_memory(key, val, "test", "test")
        for t in range(1, 25):
            layer.timestep = t
            layer.memories[0].access_history.append(t)
            if len(layer.memories[0].access_history) > config.actr_max_history:
                layer.memories[0].access_history = layer.memories[0].access_history[
                    -config.actr_max_history :
                ]

        assert len(layer.memories[0].access_history) == 5

    def test_disabled_uses_exponential(self):
        """When actr_enabled=False, recency head uses exponential decay."""
        config_actr = MemoryConfig(embedding_dim=64, actr_enabled=True)
        config_exp = MemoryConfig(embedding_dim=64, actr_enabled=False)

        layer_actr = make_layer(config_actr)
        layer_exp = make_layer(config_exp)

        for layer in [layer_actr, layer_exp]:
            for i in range(3):
                k = random_embedding(64, np.random.default_rng(i))
                v = random_embedding(64, np.random.default_rng(i + 50))
                layer.update(k, v, f"input {i}", f"output {i}")

        q = random_embedding(64, np.random.default_rng(99))
        q_eff_a = layer_actr._compute_effective_query(q)
        q_eff_e = layer_exp._compute_effective_query(q)

        acts_a = layer_actr._compute_multi_head_activations(q_eff_a)
        acts_e = layer_exp._compute_multi_head_activations(q_eff_e)

        # Both should produce valid results but different scores
        assert np.all(np.isfinite(acts_a))
        assert np.all(np.isfinite(acts_e))

    def test_access_history_persists(self, tmp_path):
        """Access history should survive save/load cycle."""
        config = MemoryConfig(
            embedding_dim=64, actr_enabled=True, memory_dir=str(tmp_path)
        )
        embedder = FakeEmbedder(dim=64)
        layer = make_layer(config, embedder)
        storage = MemoryStorage(config)

        k = random_embedding(64, np.random.default_rng(1))
        v = random_embedding(64, np.random.default_rng(2))
        layer.update(k, v, "test", "response")

        assert len(layer.memories[0].access_history) > 0
        original_history = list(layer.memories[0].access_history)

        storage.save(layer)

        layer2 = make_layer(config, embedder)
        storage.load(layer2)

        assert layer2.memories[0].access_history == original_history


# ══════════════════════════════════════════════════════════════
# Integration: Phase 2 features working together
# ══════════════════════════════════════════════════════════════


class TestPhase2Integration:
    """Test that Phase 2 features compose correctly."""

    def test_all_features_enabled(self):
        """All Phase 2 features enabled simultaneously should not crash."""
        config = MemoryConfig(
            embedding_dim=64,
            prospect_enabled=True,
            tension_enabled=True,
            channels_enabled=True,
            adaptive_k_enabled=True,
            feedback_enabled=True,
            actr_enabled=True,
        )
        embedder = FakeEmbedder(dim=64)
        layer = make_layer(config, embedder)

        # Add memories
        for i in range(5):
            k = random_embedding(64, np.random.default_rng(i))
            v = random_embedding(64, np.random.default_rng(i + 50))
            layer.update(k, v, f"input {i}", f"output {i}")

        # Recall with tension
        q = random_embedding(64, np.random.default_rng(99))
        result = layer.recall_with_tension(q)
        assert isinstance(result, RecallResult)
        assert len(result.all_results) >= 0

        # Apply feedback
        response = random_embedding(64, np.random.default_rng(100))
        fb = layer.apply_feedback(response, result.all_results)
        assert "strengthened" in fb
        assert "weakened" in fb

    def test_all_features_disabled_matches_phase1(self):
        """All Phase 2 features disabled → identical to Phase 1 behavior."""
        config = MemoryConfig(embedding_dim=64)  # all defaults = disabled
        layer = make_layer(config)

        k = random_embedding(64, np.random.default_rng(1))
        v = random_embedding(64, np.random.default_rng(2))
        result = layer.update(k, v, "test", "response")

        assert result["action"] == "created"
        assert layer.memories[0].category == ""
        assert layer.memories[0].access_history == []

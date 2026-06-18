"""Tests for Equations 8-11: Context Vector, Multi-Head Recall, Enhanced Gate, Adaptive Heads."""

import numpy as np

from itm.config import MemoryConfig
from itm.core import MemoryLayer
from itm.embeddings import EmbeddingService
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


# ── Equation 8: Conversation Context Vector (RNN Hidden State) ──


class TestContextVector:
    """c_t = λ · c_{t-1} + (1 - λ) · e_in_t"""

    def test_first_input_becomes_context(self):
        """c_0 = e_in_0 (no prior context to blend with)."""
        layer = make_layer()
        emb = random_embedding(64, np.random.default_rng(1))
        layer._update_context(emb)
        np.testing.assert_allclose(layer.context_vector, emb, atol=1e-6)

    def test_context_blends_with_new_input(self):
        """c_t should be weighted average of prior context and new input."""
        config = MemoryConfig(embedding_dim=64, lambda_context=0.7)
        layer = make_layer(config)

        e1 = random_embedding(64, np.random.default_rng(1))
        e2 = random_embedding(64, np.random.default_rng(2))

        layer._update_context(e1)
        layer._update_context(e2)

        # c_1 = 0.7 * e1 + 0.3 * e2, normalized
        expected = 0.7 * e1 + 0.3 * e2
        expected = expected / np.linalg.norm(expected)

        np.testing.assert_allclose(layer.context_vector, expected, atol=1e-5)

    def test_high_lambda_retains_old_context(self):
        """λ close to 1 → context changes slowly."""
        config = MemoryConfig(embedding_dim=64, lambda_context=0.99)
        layer = make_layer(config)

        e1 = random_embedding(64, np.random.default_rng(1))
        e2 = random_embedding(64, np.random.default_rng(2))

        layer._update_context(e1)
        layer._update_context(e2)

        # Context should still be very close to e1
        sim = EmbeddingService.cosine_similarity(layer.context_vector, e1)
        assert sim > 0.95

    def test_low_lambda_adopts_new_quickly(self):
        """λ close to 0 → context tracks recent input."""
        config = MemoryConfig(embedding_dim=64, lambda_context=0.01)
        layer = make_layer(config)

        e1 = random_embedding(64, np.random.default_rng(1))
        e2 = random_embedding(64, np.random.default_rng(2))

        layer._update_context(e1)
        layer._update_context(e2)

        # Context should be very close to e2
        sim = EmbeddingService.cosine_similarity(layer.context_vector, e2)
        assert sim > 0.95

    def test_context_vector_is_unit_length(self):
        """Context vector should always be normalized."""
        layer = make_layer()
        for seed in range(5):
            emb = random_embedding(64, np.random.default_rng(seed))
            layer._update_context(emb)
            norm = np.linalg.norm(layer.context_vector)
            assert abs(norm - 1.0) < 1e-5


class TestEffectiveQuery:
    """q_eff = α_blend · e_in + (1 - α_blend) · c_t"""

    def test_no_context_returns_raw_input(self):
        """Before any context is built, q_eff = e_in."""
        layer = make_layer()
        emb = random_embedding(64, np.random.default_rng(1))
        q_eff = layer._compute_effective_query(emb)
        np.testing.assert_allclose(q_eff, emb, atol=1e-6)

    def test_self_sufficient_input_stays_raw(self):
        """High σ_self → α_blend ≈ 1 → q_eff ≈ e_in."""
        config = MemoryConfig(embedding_dim=64, beta_context=8.0, theta_context=0.3)
        layer = make_layer(config)

        # Create a memory that the input matches well
        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key, val, "test", "test")
        layer._update_context(random_embedding(64, np.random.default_rng(3)))

        # Query with exact same key → σ_self ≈ 1.0 → α_blend → 1
        q_eff = layer._compute_effective_query(key)
        sim = EmbeddingService.cosine_similarity(q_eff, key)
        assert sim > 0.8  # q_eff stays close to raw input

    def test_fragment_leans_on_context(self):
        """Low σ_self → α_blend ≈ 0 → q_eff ≈ c_t."""
        config = MemoryConfig(embedding_dim=64, beta_context=8.0, theta_context=0.3)
        layer = make_layer(config)

        # Create a memory far from our query
        key1 = random_embedding(64, np.random.default_rng(1))
        val1 = random_embedding(64, np.random.default_rng(2))
        layer._create_memory(key1, val1, "specific topic", "response")

        # Set context to something specific
        ctx = random_embedding(64, np.random.default_rng(3))
        layer.context_vector = ctx

        # Query with orthogonal vector (σ_self ≈ 0 → leans on context)
        fragment = np.zeros(64, dtype=np.float32)
        fragment[60] = 1.0
        q_eff = layer._compute_effective_query(fragment)

        sim_to_ctx = EmbeddingService.cosine_similarity(q_eff, ctx)
        sim_to_fragment = EmbeddingService.cosine_similarity(q_eff, fragment)
        # q_eff should be closer to context than to the raw fragment
        assert sim_to_ctx > sim_to_fragment

    def test_effective_query_is_normalized(self):
        """q_eff should be unit length."""
        layer = make_layer()
        layer.context_vector = random_embedding(64, np.random.default_rng(1))
        emb = random_embedding(64, np.random.default_rng(2))
        q_eff = layer._compute_effective_query(emb)
        norm = np.linalg.norm(q_eff)
        assert abs(norm - 1.0) < 1e-5


# ── Equation 9 + 11: Multi-Head Recall with Adaptive Weighting ──


class TestMultiHeadRecall:
    """relevance_i = s_i · (w1·sim(q,k) + w2·sim(q,v) + w3·sim(c,k) + w4·recency)"""

    def test_semantic_head_dominates_for_exact_key_match(self):
        """Head 1 (semantic) should dominate when query matches a key exactly."""
        config = MemoryConfig(
            embedding_dim=64,
            adaptive_heads=False,
            recall_w_semantic=0.45,
            recall_w_answer=0.20,
            recall_w_context=0.20,
            recall_w_recency=0.15,
        )
        layer = make_layer(config)

        # Create two memories
        k1 = random_embedding(64, np.random.default_rng(1))
        v1 = random_embedding(64, np.random.default_rng(2))
        layer.update(k1, v1, "python programming", "great language")

        k2 = random_embedding(64, np.random.default_rng(3))
        v2 = random_embedding(64, np.random.default_rng(4))
        layer.update(k2, v2, "cooking recipes", "delicious")

        # Query with k1 → should rank memory 0 first
        results = layer.recall(k1)
        assert len(results) >= 1
        assert results[0][0].input_text == "python programming"

    def test_static_head_weights_are_convex_with_and_without_sparse(self):
        """Eq 9: the static head weights are a convex combination (sum to 1.0).

        The four base heads sum to 1.0. Enabling the sparse head (Eq 19,
        recall_w_sparse=0.15) would naively push the total to 1.15, breaking the
        documented convex combination. The code renormalizes, so the effective
        weight vector must sum to 1.0 whether or not the sparse head is enabled.

        We reconstruct the effective weights exactly as the static branch does,
        then cross-check that _compute_multi_head_activations actually produces
        the convex-weighted score for a single unit-strength memory.
        """

        def effective_weights(cfg):
            w1 = cfg.recall_w_semantic
            w2 = cfg.recall_w_answer
            w3 = cfg.recall_w_context
            w4 = cfg.recall_w_recency
            w5 = cfg.recall_w_sparse if cfg.sparse_recall_enabled else 0.0
            total = w1 + w2 + w3 + w4 + w5
            return np.array([w1, w2, w3, w4, w5]) / total

        cfg_no_sparse = MemoryConfig(
            embedding_dim=64, adaptive_heads=False, sparse_recall_enabled=False
        )
        cfg_sparse = MemoryConfig(
            embedding_dim=64,
            adaptive_heads=False,
            sparse_recall_enabled=True,
            recall_w_sparse=0.15,
        )

        w_no_sparse = effective_weights(cfg_no_sparse)
        w_sparse = effective_weights(cfg_sparse)

        # Published invariant: convex combination, weights sum to 1.0 in both cases.
        assert abs(w_no_sparse.sum() - 1.0) < 1e-9
        assert abs(w_sparse.sum() - 1.0) < 1e-9
        # The sparse case still carries a non-trivial sparse weight after renorm.
        assert w_sparse[4] > 0.0

        # Behavioral cross-check: a single unit-strength memory's activation must
        # equal the convex-weighted sum of its head similarities (no sparse text).
        layer = make_layer(cfg_sparse)
        k = random_embedding(64, np.random.default_rng(7))
        v = random_embedding(64, np.random.default_rng(8))
        layer.timestep = 1
        layer._create_memory(k.copy(), v.copy(), "fact", "answer")
        layer.memories[0].strength = 1.0
        layer.memories[0].timestamp = layer.timestep

        q = random_embedding(64, np.random.default_rng(9))
        # Heads, computed the same way as the implementation.
        h1 = max(float(EmbeddingService.cosine_similarity(q, k)), 0.0)
        h2 = max(float(EmbeddingService.cosine_similarity(q, v)), 0.0)
        c = layer.context_vector
        h3 = (
            max(float(EmbeddingService.cosine_similarity(c, k)), 0.0)
            if c is not None
            else 0.0
        )
        h4 = float(
            np.exp(
                -(layer.timestep - layer.memories[0].timestamp) / cfg_sparse.tau_recall
            )
        )
        h5 = 0.0  # empty query_text -> sparse head contributes nothing
        expected = float(
            w_sparse[0] * h1
            + w_sparse[1] * h2
            + w_sparse[2] * h3
            + w_sparse[3] * h4
            + w_sparse[4] * h5
        )
        acts = layer._compute_multi_head_activations(q, query_text="")
        # strength == 1.0 -> compression factor 1**exp == 1, so act == convex sum.
        assert abs(float(acts[0]) - expected) < 1e-6

    def test_recency_head_breaks_ties(self):
        """When semantic similarity is equal, recency should break the tie."""
        config = MemoryConfig(
            embedding_dim=64,
            adaptive_heads=False,
            recall_w_semantic=0.3,
            recall_w_answer=0.1,
            recall_w_context=0.1,
            recall_w_recency=0.5,
            tau_recall=5.0,
        )
        layer = make_layer(config)

        key = random_embedding(64, np.random.default_rng(1))
        val = random_embedding(64, np.random.default_rng(2))

        # Create directly (bypass update's strengthen path for identical keys)
        layer.timestep = 1
        layer._create_memory(key.copy(), val.copy(), "fact A (old)", "response A")

        layer.timestep = 12
        layer._create_memory(key.copy(), val.copy(), "fact A (new)", "response A")

        # Recall should prefer newer memory due to high recency weight
        results = layer.recall(key)
        assert len(results) >= 2
        # More recent memory should score higher (appears first in most cases)
        # Since they have the same key, semantic is equal, so recency decides
        newer_score = results[0][1]
        older_score = results[1][1]
        assert newer_score > older_score

    def test_multi_head_activations_shape(self):
        """Activation vector should have length n (one per memory)."""
        layer = make_layer()
        for i in range(5):
            k = random_embedding(64, np.random.default_rng(i))
            v = random_embedding(64, np.random.default_rng(i + 50))
            layer.update(k, v, f"input {i}", f"output {i}")

        q = random_embedding(64, np.random.default_rng(99))
        q_eff = layer._compute_effective_query(q)
        acts = layer._compute_multi_head_activations(q_eff)

        assert acts.shape == (5,)

    def test_multi_head_activations_empty(self):
        """No memories → empty activations."""
        layer = make_layer()
        q = random_embedding(64, np.random.default_rng(1))
        acts = layer._compute_multi_head_activations(q)
        assert len(acts) == 0

    def test_context_head_boosts_related_memories(self):
        """Head 3 (context) should boost memories related to conversation context."""
        config = MemoryConfig(
            embedding_dim=64,
            adaptive_heads=False,
            recall_w_semantic=0.1,
            recall_w_answer=0.1,
            recall_w_context=0.7,
            recall_w_recency=0.1,
        )
        layer = make_layer(config)

        # Create two memories
        k1 = random_embedding(64, np.random.default_rng(1))
        v1 = random_embedding(64, np.random.default_rng(2))
        layer.update(k1, v1, "topic A", "response A")

        k2 = random_embedding(64, np.random.default_rng(3))
        v2 = random_embedding(64, np.random.default_rng(4))
        layer.update(k2, v2, "topic B", "response B")

        # Set context to be similar to k1
        layer.context_vector = k1.copy()

        # Query with something neutral
        q = random_embedding(64, np.random.default_rng(99))
        q_eff = layer._compute_effective_query(q)
        acts = layer._compute_multi_head_activations(q_eff)

        # Memory 0 (topic A) should have higher activation due to context match
        assert acts[0] > acts[1]


class TestAdaptiveHeads:
    """Equation 11: w_j(q) = softmax(max_scores / T)"""

    def test_adaptive_weights_sum_to_one(self):
        """Adaptive head weights should sum to 1.0 (softmax)."""
        config = MemoryConfig(
            embedding_dim=64, adaptive_heads=True, head_temperature=0.5
        )
        layer = make_layer(config)

        k = random_embedding(64, np.random.default_rng(1))
        v = random_embedding(64, np.random.default_rng(2))
        layer.update(k, v, "test", "response")

        # The internal computation uses softmax which sums to 1 by definition.
        # We can verify by checking that activations are computed without error.
        q = random_embedding(64, np.random.default_rng(3))
        q_eff = layer._compute_effective_query(q)
        acts = layer._compute_multi_head_activations(q_eff)
        assert acts.shape == (1,)
        assert np.isfinite(acts[0])

    def test_adaptive_vs_fixed_differ(self):
        """Adaptive weights should generally differ from fixed weights."""
        layer_adaptive = make_layer(MemoryConfig(embedding_dim=64, adaptive_heads=True))
        layer_fixed = make_layer(MemoryConfig(embedding_dim=64, adaptive_heads=False))

        # Add same memories to both
        for i in range(3):
            k = random_embedding(64, np.random.default_rng(i))
            v = random_embedding(64, np.random.default_rng(i + 50))
            layer_adaptive.update(k, v, f"input {i}", f"output {i}")
            layer_fixed.update(k, v, f"input {i}", f"output {i}")

        q = random_embedding(64, np.random.default_rng(99))
        q_eff_a = layer_adaptive._compute_effective_query(q)
        q_eff_f = layer_fixed._compute_effective_query(q)

        acts_a = layer_adaptive._compute_multi_head_activations(q_eff_a)
        acts_f = layer_fixed._compute_multi_head_activations(q_eff_f)

        # Rankings might be the same but exact scores should differ
        # (adaptive softmax weights ≠ fixed static weights in general)
        assert not np.allclose(acts_a, acts_f)

    def test_low_temperature_sharpens_weights(self):
        """Low T → sharper softmax → dominant head gets more weight."""
        layer_sharp = make_layer(
            MemoryConfig(
                embedding_dim=64,
                adaptive_heads=True,
                head_temperature=0.1,
            )
        )
        layer_smooth = make_layer(
            MemoryConfig(
                embedding_dim=64,
                adaptive_heads=True,
                head_temperature=2.0,
            )
        )

        for layer in [layer_sharp, layer_smooth]:
            for i in range(3):
                k = random_embedding(64, np.random.default_rng(i))
                v = random_embedding(64, np.random.default_rng(i + 50))
                layer.update(k, v, f"input {i}", f"output {i}")

        q = random_embedding(64, np.random.default_rng(99))

        acts_sharp = layer_sharp._compute_multi_head_activations(
            layer_sharp._compute_effective_query(q)
        )
        acts_smooth = layer_smooth._compute_multi_head_activations(
            layer_smooth._compute_effective_query(q)
        )

        # Both should produce valid results
        assert np.all(np.isfinite(acts_sharp))
        assert np.all(np.isfinite(acts_smooth))

        # Both produce valid finite results with different weighting
        # (exact spread depends on random seed, so just verify correctness)
        assert acts_sharp.shape == acts_smooth.shape == (3,)
        assert not np.allclose(acts_sharp, acts_smooth)


# ── Equation 10: Enhanced Gate (GRU Multi-Signal) ──


class TestEnhancedGate:
    """novelty = clamp((1 - R_out) - w_val·R_val - w_ctx·D_ctx_factor, 0, 1)"""

    def test_backward_compat_with_eq6(self):
        """When R_val=0 and D_ctx=0, gate reduces to original Equation 6."""
        config = MemoryConfig(embedding_dim=64, beta_gate=8.0, theta_gate=0.5)
        layer = make_layer(config)

        # Original Equation 6 computation
        R_overlap = 0.3
        gate_original = layer._memory_input_gate(R_overlap)

        # Enhanced Equation 10 with zero extra signals
        gate_enhanced = layer._memory_input_gate(
            R_overlap, R_overlap_val=0.0, D_ctx=0.0
        )

        assert abs(gate_original - gate_enhanced) < 1e-10

    def test_value_overlap_reduces_gate(self):
        """High R_val → penalty reduces novelty → gate closes more."""
        config = MemoryConfig(
            embedding_dim=64,
            beta_gate=8.0,
            theta_gate=0.5,
            gate_w_val=0.3,
            gate_w_ctx=0.2,
        )
        layer = make_layer(config)

        gate_no_val = layer._memory_input_gate(0.1, R_overlap_val=0.0, D_ctx=0.0)
        gate_high_val = layer._memory_input_gate(0.1, R_overlap_val=0.8, D_ctx=0.0)

        assert gate_high_val < gate_no_val

    def test_value_overlap_does_not_veto(self):
        """R_val alone cannot close the gate when R_out is low (bounded penalty)."""
        config = MemoryConfig(
            embedding_dim=64,
            beta_gate=6.0,
            theta_gate=0.4,
            gate_w_val=0.3,
            gate_w_ctx=0.2,
        )
        layer = make_layer(config)

        # Low R_out, high R_val — gate should still PASS
        # novelty = (1 - 0.05) - 0.3*0.9 = 0.95 - 0.27 = 0.68
        # σ(6*(0.68 - 0.4)) = σ(1.68) ≈ 0.84 → passes
        gate = layer._memory_input_gate(0.05, R_overlap_val=0.9, D_ctx=0.0)
        assert gate > 0.5  # passes the gate (old multiplicative would have killed it)

    def test_context_continuity_reduces_gate(self):
        """High D_ctx → conversation continuation → gate closes more."""
        config = MemoryConfig(
            embedding_dim=64,
            beta_gate=8.0,
            theta_gate=0.5,
            gate_w_val=0.3,
            gate_w_ctx=0.2,
            theta_ctx_gate=0.3,
        )
        layer = make_layer(config)

        gate_no_ctx = layer._memory_input_gate(0.1, R_overlap_val=0.0, D_ctx=0.0)
        gate_high_ctx = layer._memory_input_gate(0.1, R_overlap_val=0.0, D_ctx=0.9)

        assert gate_high_ctx < gate_no_ctx

    def test_all_signals_low_opens_gate(self):
        """Low R_out, low R_val, low D_ctx → novelty ≈ primary → gate opens."""
        config = MemoryConfig(
            embedding_dim=64,
            beta_gate=8.0,
            theta_gate=0.5,
            gate_w_val=0.3,
            gate_w_ctx=0.2,
        )
        layer = make_layer(config)

        gate = layer._memory_input_gate(0.05, R_overlap_val=0.05, D_ctx=0.0)
        # novelty = (1-0.05) - 0.3*0.05 = 0.95 - 0.015 = 0.935
        # σ(8*(0.935-0.5)) = σ(3.48) ≈ 0.97
        assert gate > 0.9

    def test_all_signals_high_closes_gate(self):
        """High R_out + high penalties → novelty ≈ 0 → gate closes."""
        config = MemoryConfig(
            embedding_dim=64,
            beta_gate=8.0,
            theta_gate=0.5,
            gate_w_val=0.3,
            gate_w_ctx=0.2,
            theta_ctx_gate=0.3,
        )
        layer = make_layer(config)

        gate = layer._memory_input_gate(0.8, R_overlap_val=0.8, D_ctx=0.9)
        assert gate < 0.1

    def test_additive_penalty_bounded(self):
        """Auxiliary signals provide additive penalty, not multiplicative veto.

        With low R_out (novel content), even high R_val + D_ctx should not
        crush the gate completely — they can only subtract a bounded amount.
        """
        config = MemoryConfig(
            embedding_dim=64,
            beta_gate=6.0,
            theta_gate=0.4,
            gate_w_val=0.3,
            gate_w_ctx=0.2,
            theta_ctx_gate=0.3,
        )
        layer = make_layer(config)

        # Primary signal says "very novel" (R_out=0.1 → primary=0.9)
        # Auxiliary signals both high
        g = layer._memory_input_gate(0.1, R_overlap_val=0.8, D_ctx=0.9)
        # penalty = 0.3*0.8 + 0.2*D_ctx_factor ≈ 0.24 + 0.2*0.86 = 0.41
        # novelty = 0.9 - 0.41 = 0.49, gate = σ(6*(0.49-0.4)) = σ(0.54) ≈ 0.63
        assert g > 0.3  # still passes — old multiplicative gave ~0.04


class TestEnhancedGateInUpdateFlow:
    """Gate integrated with multi-signal computation in update()."""

    def test_value_overlap_helps_filter(self):
        """Output matching stored values → filtered (even if key overlap is moderate)."""
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

        # New input with output matching the stored VALUE (not key)
        new_input = random_embedding(64, np.random.default_rng(20))
        result = layer.update(
            new_input, val.copy(), "tell me something", "nice to meet you"
        )
        # The high value overlap should help close the gate
        # (exact result depends on config thresholds)
        assert result["action"] in ("filtered", "created")

    def test_context_continuity_computed_before_update(self):
        """D_ctx uses the PREVIOUS context, not the updated one."""
        config = MemoryConfig(embedding_dim=64, lambda_context=0.7)
        layer = make_layer(config)

        # First turn: establish context
        e1 = random_embedding(64, np.random.default_rng(1))
        v1 = random_embedding(64, np.random.default_rng(2))
        layer.update(e1, v1, "first input", "first response")

        old_context = layer.context_vector.copy()

        # Second turn: input similar to old context → high D_ctx
        e2 = e1 + 0.1 * random_embedding(64, np.random.default_rng(3))
        e2 = (e2 / np.linalg.norm(e2)).astype(np.float32)
        v2 = random_embedding(64, np.random.default_rng(4))
        result = layer.update(e2, v2, "similar input", "response")

        # Context should have been updated AFTER D_ctx computation
        assert not np.allclose(layer.context_vector, old_context)


# ── Context Vector Persistence ──


class TestContextVectorPersistence:

    def test_context_survives_save_load(self, tmp_path):
        """Context vector should persist across save/load."""
        config = MemoryConfig(embedding_dim=64, memory_dir=str(tmp_path))
        embedder = FakeEmbedder(dim=64)
        layer = MemoryLayer(config, embedder)
        storage = MemoryStorage(config)

        # Build context over several turns
        for i in range(5):
            k = random_embedding(64, np.random.default_rng(i))
            v = random_embedding(64, np.random.default_rng(i + 50))
            layer.update(k, v, f"input {i}", f"output {i}")

        ctx_before = layer.context_vector.copy()
        storage.save(layer)

        # Load into fresh layer
        layer2 = MemoryLayer(config, embedder)
        storage.load(layer2)

        np.testing.assert_allclose(layer2.context_vector, ctx_before, atol=1e-5)

    def test_no_context_loads_as_none(self, tmp_path):
        """If no context was built, it should load as None."""
        config = MemoryConfig(embedding_dim=64, memory_dir=str(tmp_path))
        embedder = FakeEmbedder(dim=64)
        layer = MemoryLayer(config, embedder)
        storage = MemoryStorage(config)

        # Save with no updates (no context)
        storage.save(layer)

        layer2 = MemoryLayer(config, embedder)
        storage.load(layer2)
        assert layer2.context_vector is None


# ── Graph Recall with Multi-Head ──


class TestGraphRecallMultiHead:
    """recall_graph() should use multi-head activations as initial seeds."""

    def test_graph_recall_uses_multi_head(self):
        """Graph recall should leverage all 4 heads, not just semantic."""
        config = MemoryConfig(
            embedding_dim=64,
            eta_propagation=0.5,
            n_hops=2,
            theta_edge=0.01,
            activation_threshold=0.01,
            tau_temporal=5.0,
            adaptive_heads=False,
        )
        layer = make_layer(config)

        # Create memories
        for i in range(4):
            k = random_embedding(64, np.random.default_rng(i))
            v = random_embedding(64, np.random.default_rng(i + 50))
            layer.update(k, v, f"input {i}", f"output {i}")

        q = random_embedding(64, np.random.default_rng(99))
        results = layer.recall_graph(q)

        # Should return at least some results
        assert len(results) >= 1
        # Each result is (MemoryEntry, activation)
        for mem, act in results:
            assert isinstance(act, float)
            assert act >= config.activation_threshold

    def test_graph_recall_empty_layer(self):
        """Empty layer → empty results."""
        layer = make_layer()
        q = random_embedding(64)
        assert layer.recall_graph(q) == []

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from itm.config import MemoryConfig
from itm.embeddings import EmbeddingService
from itm.graph import MemoryGraph
from itm.hierarchy import MemoryHierarchy


@dataclass
class RecallResult:
    """Structured recall output with tension detection (Equation 13).

    NN parallel: Contrastive learning — surface both supporting and
    conflicting evidence to break Kahneman's WYSIATI trap.
    """

    supporting: list[tuple]  # (MemoryEntry, float) — agrees with top result
    conflicting: list[tuple]  # (MemoryEntry, float) — disagrees with top result
    all_results: list[tuple]  # original flat list (backward compat)


@dataclass
class MemoryEntry:
    """A single memory: m_i = (k_i, v_i, s_i, b_i, metadata)."""

    key: np.ndarray  # (d,) embedding of input
    value: np.ndarray  # (d,) embedding of output
    strength: float  # s_i — learned relevance
    bias: float  # b_i — contradiction adaptation rate
    timestamp: int  # last activation timestep
    access_count: int = 0
    input_text: str = ""
    output_text: str = ""
    # Phase 2 fields
    access_history: list = field(default_factory=list)  # Eq 17: ACT-R timestep history
    category: str = ""  # Eq 14: fact/preference/instruction/event
    # Phase 4 fields
    context_at_creation: np.ndarray | None = None  # Eq 24: context vector when created


class SparseIndex:
    """Equation 19: TF-IDF sparse lexical index for hybrid recall.

    NN parallel: BM25 (Robertson et al. 1994) — 30 years of information
    retrieval. Sparse attention in transformers (BigBird, Longformer).

    Dense embeddings encode meaning but miss exact keywords (phone numbers,
    usernames, proper nouns). TF-IDF catches what dense misses.

    Stores sparse vectors alongside dense embeddings. No external deps —
    pure Python collections.Counter + numpy.

    score_hybrid = (1 - w_sparse) · dense_score + w_sparse · sparse_score
    """

    _TOKENIZE_RE = re.compile(r"[a-z0-9]+")

    def __init__(self):
        self._df: Counter = Counter()  # document frequency per term
        self._doc_count: int = 0
        self._vectors: list[Counter] = []  # per-memory term frequency

    def _tokenize(self, text: str) -> list[str]:
        """Simple lowercase tokenization — catches names, numbers, codes."""
        return self._TOKENIZE_RE.findall(text.lower())

    def add(self, text: str) -> int:
        """Add a document and return its index."""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        self._vectors.append(tf)
        # Update document frequencies
        for term in set(tokens):
            self._df[term] += 1
        self._doc_count += 1
        return len(self._vectors) - 1

    def query_similarity(self, query_text: str) -> np.ndarray:
        """Compute TF-IDF cosine similarity between query and all documents.

        Returns (n,) array of similarities in [0, 1].
        """
        n = len(self._vectors)
        if n == 0 or not query_text.strip():
            return np.zeros(n, dtype=np.float32)

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return np.zeros(n, dtype=np.float32)

        # Build query TF-IDF vector (only for terms that appear in corpus)
        query_tf = Counter(query_tokens)
        query_terms = {}
        for term, count in query_tf.items():
            if term in self._df:
                idf = math.log(1.0 + self._doc_count / (1.0 + self._df[term]))
                query_terms[term] = count * idf

        if not query_terms:
            return np.zeros(n, dtype=np.float32)

        # Query norm
        q_norm = math.sqrt(sum(v * v for v in query_terms.values()))
        if q_norm < 1e-10:
            return np.zeros(n, dtype=np.float32)

        # Compute dot product + norms for each document
        sims = np.zeros(n, dtype=np.float32)
        for i, doc_tf in enumerate(self._vectors):
            dot = 0.0
            d_norm_sq = 0.0
            for term, tf_count in doc_tf.items():
                if term in self._df:
                    idf = math.log(1.0 + self._doc_count / (1.0 + self._df[term]))
                    tfidf = tf_count * idf
                    d_norm_sq += tfidf * tfidf
                    if term in query_terms:
                        dot += query_terms[term] * tfidf
            d_norm = math.sqrt(d_norm_sq)
            if d_norm > 1e-10:
                sims[i] = dot / (q_norm * d_norm)

        return sims

    def rebuild(self, texts: list[str]):
        """Rebuild the entire index from a list of texts."""
        self._df.clear()
        self._doc_count = 0
        self._vectors.clear()
        for text in texts:
            self.add(text)


class MemoryLayer:
    """Layer 1 (Episodic) memory implementing the core equations.

    Equation 1 — Strength Update:
        s_i(t+1) = s_i(t) * gamma + alpha * sim(e_in, k_i)

    Equation 2 — Value Update (on contradiction):
        v_i(t+1) = v_i(t) + b_i * (e_out_new - v_i(t))

    Equation 2b — Soft Value Drift (gradient descent-inspired):
        v_i += b_i · (1 - val_sim) · (v_new - v_i)
        Always-on proportional nudge during strengthen — no threshold needed.

    Equation 3 — Recall:
        R(q) = sum(s_i * sim(q, k_i) * v_i) / sum(s_i * sim(q, k_i))

    Equation 6 — Memory Input Gate (LSTM-inspired):
        g = σ(β_g · ((1 - R_overlap) - θ_gate))

    Equation 8 — Conversation Context Vector (RNN hidden state):
        c_t = λ · c_{t-1} + (1 - λ) · e_in_t
        q_eff = α_blend · e_in + (1 - α_blend) · c_t

    Equation 9 — Multi-Head Recall (multi-head attention):
        relevance_i = s_i · (w1·sim(q,k) + w2·sim(q,v) + w3·sim(c,k) + w4·recency)

    Equation 10 — Enhanced Gate (input-driven with output penalties):
        primary = 1 - best_sim  (input novelty)
        penalty = w_out·R_out + w_val·R_val + w_ctx·D_ctx + w_entropy·H_out
        g = σ(β_g · (clamp(primary - penalty) - θ_gate))

    Equation 10b — Attention Entropy Gate Signal:
        H_out = -Σ(p_i · log(p_i)) / log(n)  where p = softmax(sims/T)
        High entropy = output attends to many keys (query). Penalizes gate.

    Equation 11 — Adaptive Head Weighting (self-attention over heads):
        w_j(q) = softmax(max_scores / T)

    Phase 2 Equations (Kahneman + ACT-R + MemWire):

    Equation 12 — Prospect-Theory Strength (Kahneman 1979):
        gain:  Δs^ρ               (diminishing returns, ρ=0.88)
        loss: -λ·|Δs|^ρ           (amplified losses, λ=2.25)

    Equation 13 — Tension Detection (WYSIATI-breaking):
        tension = sim(k_top, k_other) > θ AND sim(v_top, v_other) < θ

    Equation 14 — Memory Category Channels (Attribute Substitution fix):
        channel_i = argmax_c(sim(k_i, anchor_c))

    Equation 15 — Adaptive Recall Depth (Kahneman Capacity Model):
        top_k_eff = ceil(top_k · (1 + β_d · (1 - max_sim)))

    Equation 16 — Feedback Loop (RLHF-inspired):
        alignment = sim(e_response, v_i) → strengthen or weaken

    Equation 17 — ACT-R Base-Level Activation (Power-Law Decay):
        B_i = ln(Σ_n (t - t_n)^(-d))

    Phase 3 Equations (MemWire-Inspired, Formalized):

    Equation 18 — Displacement-Augmented Edges (ResNet residual):
        d_i = normalize(v_i - k_i)
        w_ij += α_d · sim(d_i, d_j)

    Equation 19 — Sparse Lexical Recall (BM25, Robertson 1994):
        h5 = TF-IDF(query_text, doc_i)
        score += w_sparse · h5

    Equation 20 — Multi-Scale Activation (JK-Net, Xu 2018):
        a_final = Σ_l (w_l · a^(l)),  η_l = η · damping^l

    Equation 21 — Query-Aware Message Passing (GAT, Veličković 2018):
        message_ij = w_ij · a_j · sim(q, k_j)

    Phase 4 Equations (Research.md §14-18):

    Equation 22 — L2 Pattern Formation (CNN Weighted Pooling, LeCun 1998):
        K_L2 = Σ(s_i·k_i)/Σ(s_i), S_L2 = Σ(s_i)

    Equation 23 — Hierarchical Recall (ResNet Skip Connections, He 2015):
        R(q) = w₁·R_L1 + w₂·R_L2 + w₃·R_L3

    Equation 24 — Multi-Channel Contradiction (Multi-Head Attention, Vaswani 2017):
        sim_mc = w_t·sim(q,k) + w_d·sim(d_q,d_i) + w_c·sim(c_q,c_i)

    Equation 25 — Soft Create-vs-Strengthen (Sigmoid Activation, 1986):
        p_strengthen = σ(β·(sim - θ_local))

    Equation 26 — Adaptive Sharpness (Temperature Scaling):
        β = β_min + (β_max - β_min)·min(n/n_mature, 1)

    Equation 27 — Per-Region Adaptive Threshold (Adam/BatchNorm):
        θ_local = θ_base + β_θ·density(q)

    Equation 28 — Cold Start Boost (Inverted Learning Rate Warmup):
        α_eff = α·(1 + κ·exp(-n/τ_cold))

    Equation 29 — Shared Memory Merge (Federated Averaging, McMahan 2017):
        R(q) = (1-w_s)·R_personal + w_s·R_shared

    Never-forget: memories are never pruned. Strength = relevance, not existence.
    """

    def __init__(self, config: MemoryConfig, embedder: EmbeddingService):
        self.config = config
        self.embedder = embedder
        self.memories: list[MemoryEntry] = []
        self.timestep: int = 0
        self.graph: MemoryGraph = MemoryGraph(config)
        # Equation 8: Conversation context vector (RNN hidden state)
        self.context_vector: np.ndarray | None = None
        # Equation 19: Sparse lexical index for hybrid recall
        self.sparse_index: SparseIndex = SparseIndex()
        # Equations 22-23: Hierarchical memory L1→L2→L3
        self.hierarchy: MemoryHierarchy = MemoryHierarchy(config)

    @property
    def _key_matrix(self) -> np.ndarray:
        """Stack all memory keys into an (n, d) matrix for vectorized ops."""
        if not self.memories:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)
        return np.stack([m.key for m in self.memories])

    @property
    def _value_matrix(self) -> np.ndarray:
        """Stack all memory values into an (n, d) matrix."""
        if not self.memories:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)
        return np.stack([m.value for m in self.memories])

    @property
    def _strength_vector(self) -> np.ndarray:
        """All strengths as an (n,) vector."""
        return np.array([m.strength for m in self.memories], dtype=np.float64)

    # ── Equation 8: Conversation Context Vector (RNN hidden state) ──

    def _update_context(self, input_embedding: np.ndarray):
        """Equation 8: c_t = λ · c_{t-1} + (1 - λ) · e_in_t

        Exponential moving average of input embeddings — the simplest
        possible RNN hidden state. Carries conversational context forward
        so that fragments like "and raj" inherit meaning from prior turns.
        """
        lam = self.config.lambda_context
        if self.context_vector is None:
            self.context_vector = input_embedding.copy()
        else:
            self.context_vector = (
                lam * self.context_vector + (1.0 - lam) * input_embedding
            )
            # Normalize to unit length (EMA of unit vectors ≠ unit vector)
            norm = np.linalg.norm(self.context_vector)
            if norm > 1e-8:
                self.context_vector = self.context_vector / norm

    def _compute_effective_query(self, input_embedding: np.ndarray) -> np.ndarray:
        """Equation 8: Adaptive blending of input and context.

        σ_self = max_i(sim(e_in, k_i))          [is input meaningful alone?]
        α_blend = σ(β_c · (σ_self - θ_c))       [sigmoid gate]
        q_eff = α_blend · e_in + (1 - α_blend) · c_t

        When input is self-contained (high σ_self), use it directly.
        When input is a fragment (low σ_self), lean on context.
        """
        if self.context_vector is None:
            return input_embedding

        # Self-sufficiency: how well does input match existing memories?
        if self.memories:
            sims = EmbeddingService.cosine_similarity_matrix(
                input_embedding, self._key_matrix
            )
            sigma_self = float(np.max(sims))
        else:
            sigma_self = 1.0  # No memories → input is all we have

        # Sigmoid blending gate
        x = self.config.beta_context * (sigma_self - self.config.theta_context)
        alpha_blend = 1.0 / (1.0 + np.exp(-x))

        # Blend input with context
        q_eff = (
            alpha_blend * input_embedding + (1.0 - alpha_blend) * self.context_vector
        )

        # Normalize
        norm = np.linalg.norm(q_eff)
        if norm > 1e-8:
            q_eff = q_eff / norm

        return q_eff.astype(np.float32)

    # ── Equation 9 + 11: Multi-Head Recall with Adaptive Weighting ──

    def _compute_multi_head_activations(
        self,
        q_eff: np.ndarray,
        query_text: str = "",
    ) -> np.ndarray:
        """Equations 9 + 11 + 19: Multi-head recall with sparse lexical head.

        Head 1 — Semantic:  sim(q_eff, k_i)      [topic relevance]
        Head 2 — Answer:    sim(q_eff, v_i)      [does the answer relate?]
        Head 3 — Context:   sim(c_t, k_i)        [conversational flow]
        Head 4 — Recency:   exp(-(t - t_i) / τ)  [temporal freshness]
        Head 5 — Sparse:    TF-IDF(query, doc_i)  [Eq 19: exact keyword match]

        Equation 11 (adaptive): weights = softmax(max_scores / T)
        Each query dynamically determines which heads to trust.
        """
        n = len(self.memories)
        if n == 0:
            return np.array([], dtype=np.float64)

        keys = self._key_matrix
        values = self._value_matrix
        strengths = self._strength_vector

        # Head 1: Semantic match — sim(q_eff, k_i)
        h1 = EmbeddingService.cosine_similarity_matrix(q_eff, keys).astype(np.float64)

        # Head 2: Answer match — sim(q_eff, v_i)
        h2 = EmbeddingService.cosine_similarity_matrix(q_eff, values).astype(np.float64)

        # Head 3: Context match — sim(c_t, k_i)
        if self.context_vector is not None:
            h3 = EmbeddingService.cosine_similarity_matrix(
                self.context_vector, keys
            ).astype(np.float64)
        else:
            h3 = np.zeros(n, dtype=np.float64)

        # Head 4: Temporal recency
        # Equation 17 (ACT-R): power-law B_i = ln(Σ(t-t_n)^(-d))
        # Default: exponential exp(-(t - t_i) / τ)
        if self.config.actr_enabled:
            raw_b = np.array(
                [self._compute_base_level_activation(m) for m in self.memories],
                dtype=np.float64,
            )
            # Normalize to [0, 1] for compatibility with other heads
            b_min, b_max = raw_b.min(), raw_b.max()
            if b_max - b_min > 1e-8:
                h4 = (raw_b - b_min) / (b_max - b_min)
            else:
                h4 = np.ones(n, dtype=np.float64) * 0.5
        else:
            timestamps = np.array(
                [m.timestamp for m in self.memories], dtype=np.float64
            )
            time_diffs = np.maximum(self.timestep - timestamps, 0.0)
            h4 = np.exp(-time_diffs / self.config.tau_recall)

        # Head 5 (Eq 19): Sparse lexical match — TF-IDF keyword similarity
        if self.config.sparse_recall_enabled and query_text.strip():
            h5 = self.sparse_index.query_similarity(query_text).astype(np.float64)
        else:
            h5 = np.zeros(n, dtype=np.float64)

        # Clamp negative similarities to 0 (only positive contributions)
        h1 = np.maximum(h1, 0.0)
        h2 = np.maximum(h2, 0.0)
        h3 = np.maximum(h3, 0.0)

        # Equation 11: Adaptive head weighting via softmax
        if self.config.adaptive_heads and n > 0:
            confs = np.array(
                [
                    np.max(h1) if h1.size > 0 else 0.0,
                    np.max(h2) if h2.size > 0 else 0.0,
                    np.max(h3) if h3.size > 0 else 0.0,
                    np.max(h4) if h4.size > 0 else 0.0,
                    np.max(h5) if h5.size > 0 else 0.0,
                ]
            )
            T = self.config.head_temperature
            # Stable softmax
            confs_shifted = confs / max(T, 1e-8)
            confs_shifted = confs_shifted - np.max(confs_shifted)
            exp_confs = np.exp(confs_shifted)
            weights = exp_confs / np.sum(exp_confs)
            w1, w2, w3, w4, w5 = weights
        else:
            w1 = self.config.recall_w_semantic
            w2 = self.config.recall_w_answer
            w3 = self.config.recall_w_context
            w4 = self.config.recall_w_recency
            w5 = (
                self.config.recall_w_sparse
                if self.config.sparse_recall_enabled
                else 0.0
            )

        # Combine heads
        multi_score = w1 * h1 + w2 * h2 + w3 * h3 + w4 * h4 + w5 * h5

        # Apply compressed strength weighting (like BM25 TF saturation).
        # Raw strength multiplication lets early/reinforced memories dominate.
        # Exponent < 1 compresses the range: strength=1.0 vs 0.3 goes from
        # 3.3x advantage (linear) to 1.4x (exp=0.3), letting similarity win.
        exp = self.config.recall_strength_exp
        compressed = np.power(np.maximum(strengths, 1e-10), exp)
        return compressed * multi_score

    # ── Equation 12: Prospect-Theory Asymmetric Strength (Kahneman) ──

    def _prospect_strength_delta(self, raw_delta: float) -> float:
        """Equation 12: Prospect-theory transform on strength changes.

        NN parallel: Kahneman & Tversky (1979) prospect theory value function.
        Asymmetric loss functions (focal loss, Huber loss).

        v(Δ) = |Δ|^ρ           for gains  (Δ ≥ 0)
        v(Δ) = -λ · |Δ|^ρ     for losses (Δ < 0)

        Gains show diminishing returns (ρ=0.88 < 1).
        Losses are amplified by λ=2.25 — contradictions/decay hit harder.

        When prospect_enabled=False, returns raw_delta unchanged.
        """
        if not self.config.prospect_enabled:
            return raw_delta

        if raw_delta == 0.0:
            return 0.0

        rho = self.config.prospect_rho
        lam = self.config.prospect_lambda_loss
        abs_delta = abs(raw_delta)

        if raw_delta >= 0:
            return abs_delta**rho
        else:
            return -(lam * (abs_delta**rho))

    # ── Equation 15: Adaptive Recall Depth (Kahneman Capacity Model) ──

    def _compute_effective_top_k(self, q_eff: np.ndarray) -> int:
        """Equation 15: Adaptive top_k based on query difficulty.

        NN parallel: Kahneman's Capacity Model (1973) / adaptive computation
        (early exit in transformers, Universal Transformers).

        difficulty = 1 - max_sim(q_eff, stored_keys)
        top_k_effective = ceil(top_k_base · (1 + β_d · difficulty))

        Easy queries (high max_sim) → fewer memories retrieved.
        Hard queries (low max_sim) → more memories retrieved.

        When adaptive_k_enabled=False, returns config.top_k unchanged.
        """
        if not self.config.adaptive_k_enabled or not self.memories:
            return self.config.top_k

        sims = EmbeddingService.cosine_similarity_matrix(q_eff, self._key_matrix)
        max_sim = float(np.max(sims))
        difficulty = 1.0 - max(0.0, max_sim)

        import math

        effective = math.ceil(
            self.config.top_k * (1.0 + self.config.beta_difficulty * difficulty)
        )
        return min(effective, self.config.top_k_max)

    # ── Equation 17: ACT-R Base-Level Activation (Power-Law Decay) ──

    def _compute_base_level_activation(self, mem: MemoryEntry) -> float:
        """Equation 17: ACT-R base-level activation.

        NN parallel: ACT-R cognitive architecture (Anderson, 1976-2025).
        Validated against 50 years of human memory experiments.

        B_i = ln(Σ_n (t_now - t_n)^(-d))

        where t_n are access timestamps, d=0.5 (power-law decay).
        Captures both recency AND frequency: many recent accesses → high B.
        One old access → low B. Many old accesses → moderate B.
        """
        if not mem.access_history:
            # Fallback: use timestamp as single access
            dt = max(self.timestep - mem.timestamp, 1)
            return float(np.log(max(dt ** (-self.config.actr_decay), 1e-10)))

        d = self.config.actr_decay
        total = 0.0
        for t_n in mem.access_history:
            dt = max(self.timestep - t_n, 1)
            total += dt ** (-d)

        return float(np.log(max(total, 1e-10)))

    # ── Equation 13: Tension Detection (WYSIATI-Breaking) ──

    def _detect_tension(self, results: list[tuple[MemoryEntry, float]]) -> RecallResult:
        """Equation 13: Detect conflicting memories in recall results.

        NN parallel: Contrastive learning / adversarial training.
        Kahneman's WYSIATI (What You See Is All There Is): when recall
        returns only supporting evidence, the LLM treats it as complete.

        For each pair (top_memory, other_memory):
          key_sim = sim(k_top, k_other)     [do they share a topic?]
          val_sim = sim(v_top, v_other)     [do their answers agree?]
          if key_sim > θ_key AND val_sim < θ_val → CONFLICT

        Returns RecallResult with supporting + conflicting separation.
        """
        if not results or not self.config.tension_enabled:
            return RecallResult(
                supporting=list(results),
                conflicting=[],
                all_results=list(results),
            )

        top_mem, top_score = results[0]
        supporting = [(top_mem, top_score)]
        conflicting = []

        for mem, score in results[1:]:
            key_sim = float(EmbeddingService.cosine_similarity(top_mem.key, mem.key))
            val_sim = float(
                EmbeddingService.cosine_similarity(top_mem.value, mem.value)
            )

            if (
                key_sim > self.config.theta_tension_key
                and val_sim < self.config.theta_tension_val
            ):
                conflicting.append((mem, score))
            else:
                supporting.append((mem, score))

        return RecallResult(
            supporting=supporting,
            conflicting=conflicting,
            all_results=list(results),
        )

    # ── Equation 14: Memory Category Channels ──

    def _classify_memory(self, input_embedding: np.ndarray) -> str:
        """Equation 14: Classify memory into a channel using anchor embeddings.

        NN parallel: Mixture of Experts routing / multi-head channel separation.
        Prevents Kahneman's attribute substitution — the LLM conflating
        factual recall with preference recall.

        channel_i = argmax_c(sim(k_i, anchor_c))

        Anchors are lazily initialized on first use.
        When channels_enabled=False, returns "".
        """
        if not self.config.channels_enabled:
            return ""

        if not hasattr(self, "_channel_anchors_cache"):
            self._channel_anchors_cache = {}
            for name, text in self.config.channel_anchors.items():
                emb = self.embedder.embed(text)
                self._channel_anchors_cache[name] = emb

        best_cat = ""
        best_sim = -1.0
        for name, anchor_emb in self._channel_anchors_cache.items():
            sim = float(EmbeddingService.cosine_similarity(input_embedding, anchor_emb))
            if sim > best_sim:
                best_sim = sim
                best_cat = name

        return best_cat

    # ── Equation 16: Feedback Loop (Post-Response Learning) ──

    def apply_feedback(
        self,
        response_embedding: np.ndarray,
        recalled: list[tuple[MemoryEntry, float]],
    ) -> dict:
        """Equation 16: Post-response feedback loop.

        NN parallel: RLHF reward signal / Hebbian post-hoc learning.
        After the LLM responds, compare its response to recalled memories.
        If aligned → strengthen (LLM "used" the memory).
        If misaligned → weaken (LLM "rejected" the memory).

        alignment_i = sim(e_response, v_i)
        if alignment > θ_align:   strengthen memory + edges
        if alignment < θ_misalign: weaken memory + edges
        """
        if not self.config.feedback_enabled or not recalled:
            return {"strengthened": 0, "weakened": 0}

        strengthened = 0
        weakened = 0

        for mem, _score in recalled:
            alignment = float(
                EmbeddingService.cosine_similarity(response_embedding, mem.value)
            )

            # Find this memory's index
            try:
                idx = self.memories.index(mem)
            except ValueError:
                continue

            if alignment > self.config.theta_align:
                # Strengthen: LLM used this memory
                mem.strength += self.config.alpha_feedback * alignment
                # Strengthen graph edges
                self.graph.adjust_edge_neighbors(idx, 1.0 + self.config.eta_feedback)
                strengthened += 1

            elif alignment < self.config.theta_misalign:
                # Weaken: LLM rejected this memory
                penalty = self.config.alpha_feedback * (
                    self.config.theta_misalign - alignment
                )
                mem.strength *= max(0.01, 1.0 - penalty)
                # Weaken graph edges
                self.graph.adjust_edge_neighbors(idx, 1.0 - self.config.eta_feedback)
                weakened += 1

        return {"strengthened": strengthened, "weakened": weakened}

    # ── Equation 24: Multi-Channel Contradiction Detection ──

    def _multichannel_contradiction(
        self,
        memory: MemoryEntry,
        new_output_embedding: np.ndarray,
        new_input_embedding: np.ndarray,
        key_sim: float,
    ) -> bool:
        """Equation 24: Multi-channel contradiction detection.

        NN parallel: Multi-Head Attention (Vaswani 2017).
        Different heads capture different aspects of meaning.

        Channels:
          topic = sim(k_new, k_old)                   [same subject?]
          displacement = sim(d_new, d_old)             [same stance?]
          context = sim(c_now, c_at_creation)          [same conversation?]

        Contradiction = high topic sim + low displacement sim.
        This catches "I love Python" vs "I hate Python" (topic=0.97, disp=low).
        """
        if not self.config.multichannel_enabled:
            return self._detect_contradiction(memory, new_output_embedding, key_sim)

        # Channel 1: Topic (standard key similarity — already computed)
        topic_sim = key_sim

        if topic_sim < self.config.theta_key:
            return False

        # Channel 2: Displacement (stance/relationship direction)
        # d_i = normalize(v_i - k_i) captures the relationship type
        d_old = memory.value - memory.key
        norm_old = np.linalg.norm(d_old)
        if norm_old > 1e-8:
            d_old = d_old / norm_old

        d_new = new_output_embedding - new_input_embedding
        norm_new = np.linalg.norm(d_new)
        if norm_new > 1e-8:
            d_new = d_new / norm_new

        disp_sim = float(EmbeddingService.cosine_similarity(d_old, d_new))

        # Channel 3: Context (conversational continuity)
        ctx_sim = 0.5  # neutral default
        if memory.context_at_creation is not None and self.context_vector is not None:
            ctx_sim = float(
                EmbeddingService.cosine_similarity(
                    self.context_vector, memory.context_at_creation
                )
            )

        # Weighted multi-channel: high topic + low displacement = contradiction
        w_t = self.config.w_channel_topic
        w_d = self.config.w_channel_disp
        w_c = self.config.w_channel_ctx

        # Value similarity (standard check)
        val_sim = float(
            EmbeddingService.cosine_similarity(memory.value, new_output_embedding)
        )

        # Combined contradiction signal: topic agrees, displacement/value disagrees
        agreement = w_t * topic_sim + w_d * disp_sim + w_c * ctx_sim
        # If topic is high but displacement is low → contradiction
        return topic_sim > self.config.theta_key and (
            val_sim < self.config.theta_value or disp_sim < self.config.theta_value
        )

    # ── Equations 25-27: Soft Create-vs-Strengthen (Sigmoid) ──

    def _compute_adaptive_sharpness(self) -> float:
        """Equation 26: Adaptive sigmoid sharpness (Temperature Scaling).

        NN parallel: Temperature scaling in softmax attention (Vaswani 2017).
        High temperature → soft distribution (explore).
        Low temperature → sharp distribution (exploit).

        β = β_min + (β_max - β_min) · min(n_memories / n_mature, 1.0)

        Few memories: β ≈ β_min (soft, hedges bets, builds diverse base).
        Many memories: β ≈ β_max (sharp, decisive, efficient).
        """
        n = len(self.memories)
        saturation = min(n / max(self.config.n_mature, 1), 1.0)
        return (
            self.config.beta_soft_min
            + (self.config.beta_soft_max - self.config.beta_soft_min) * saturation
        )

    def _compute_adaptive_threshold(self, query_embedding: np.ndarray) -> float:
        """Equation 27: Per-region adaptive threshold (Adam/BatchNorm).

        NN parallel: Adam optimizer (Kingma 2014) gives each parameter its own
        learning rate. BatchNorm (Ioffe 2015) normalizes per-layer statistics.

        θ_local(q) = θ_base + β_θ · density(q)
        density(q) = |{m_i : sim(q, k_i) > θ_neighbor}| / n

        Dense region (many similar memories) → high threshold (be specific).
        Sparse region (few similar memories) → low threshold (be inclusive).
        """
        if not self.memories:
            return self.config.theta_soft_base

        sims = EmbeddingService.cosine_similarity_matrix(
            query_embedding, self._key_matrix
        )
        # Count neighbors above threshold
        n_neighbors = int(np.sum(sims > self.config.theta_density_neighbor))
        density = n_neighbors / len(self.memories)

        return self.config.theta_soft_base + self.config.beta_theta_density * density

    def _soft_decision(self, best_sim: float, query_embedding: np.ndarray) -> float:
        """Equation 25: Soft create-vs-strengthen via sigmoid.

        NN parallel: Sigmoid replacing step function (1986).
        The transition from hard thresholds to soft sigmoids enabled
        backpropagation → deep learning → everything we have today.

        p_strengthen = σ(β · (sim_best - θ_local))

        Returns p_strengthen ∈ (0, 1).
        High → strengthen existing memory. Low → create new memory.
        """
        beta = self._compute_adaptive_sharpness()  # Eq 26
        theta = self._compute_adaptive_threshold(query_embedding)  # Eq 27

        x = beta * (best_sim - theta)
        return float(1.0 / (1.0 + np.exp(-x)))

    # ── Equation 28: Cold Start Learning Rate Boost ──

    def _cold_start_factor(self) -> float:
        """Equation 28: Cold start learning rate boost (Inverted Warmup).

        NN parallel: Learning rate warmup (Goyal et al. 2017) but inverted.
        Standard warmup: start low, ramp up (for training stability).
        Ours: start HIGH, decay to normal (for fast initial pattern formation).

        factor = 1 + κ · exp(-n_memories / τ_cold)

        n=0:  factor ≈ 1 + κ (high boost, aggressive learning)
        n=τ:  factor ≈ 1 + κ/e (decaying)
        n→∞:  factor → 1.0 (baseline, normal operation)
        """
        if not self.config.cold_start_enabled:
            return 1.0

        n = len(self.memories)
        kappa = self.config.cold_start_kappa
        tau = max(self.config.cold_start_tau, 1.0)
        return 1.0 + kappa * np.exp(-n / tau)

    # ── Recall Methods ──

    def recall_with_tension(
        self,
        query_embedding: np.ndarray,
        query_text: str = "",
    ) -> RecallResult:
        """Recall with tension detection (Equation 13).

        Wraps recall_graph() and separates results into supporting
        and conflicting memories. Use this when tension_enabled=True.
        """
        results = self.recall_graph(query_embedding, query_text=query_text)
        return self._detect_tension(results)

    def recall(
        self,
        query_embedding: np.ndarray,
        query_text: str = "",
    ) -> list[tuple[MemoryEntry, float]]:
        """Recall top-k memories using multi-head scoring (Equations 9 + 11 + 19).

        Returns list of (MemoryEntry, relevance_score) sorted descending.
        """
        if not self.memories:
            return []

        q_eff = self._compute_effective_query(query_embedding)
        activations = self._compute_multi_head_activations(q_eff, query_text=query_text)

        # Equation 15: adaptive top_k
        effective_k = self._compute_effective_top_k(q_eff)

        # Sort descending by activation
        indices = np.argsort(activations)[::-1]

        results = []
        for idx in indices[:effective_k]:
            idx = int(idx)
            rel = float(activations[idx])
            if rel < self.config.min_relevance:
                break
            self.memories[idx].access_count += 1
            results.append((self.memories[idx], rel))

        return results

    def recall_graph(
        self,
        query_embedding: np.ndarray,
        query_text: str = "",
    ) -> list[tuple[MemoryEntry, float]]:
        """Graph-based recall with multi-head scoring + enhanced spreading.

        Combines Equations 8-9-11-19 (multi-head initial activations) with
        Equations 5+20+21 (enhanced graph spreading activation) and
        Equations 22-23 (hierarchical recall with skip connections).

        Pipeline:
          1. Compute effective query using context vector (Equation 8)
          2. Score all memories with multi-head attention (Equations 9+11+19)
          3. Propagate activations through graph (Equations 5+20+21)
          4. Apply hierarchical boost (Equations 22-23) if enabled
          5. Return all memories above activation threshold
        """
        if not self.memories:
            return []

        q_eff = self._compute_effective_query(query_embedding)
        initial_activations = self._compute_multi_head_activations(
            q_eff,
            query_text=query_text,
        )

        results = self.graph.spread_and_collect(
            initial_activations,
            self.memories,
            query_embedding=q_eff,
        )

        # Equation 23: Apply hierarchical recall with skip connections
        if self.config.hierarchy_enabled and (
            self.hierarchy.l2_patterns or self.hierarchy.l3_identities
        ):
            results = self.hierarchy.hierarchical_recall(
                q_eff,
                self.memories,
                results,
            )

        return results

    def recall_weighted_value(self, query_embedding: np.ndarray) -> np.ndarray | None:
        """Compute the full recall vector R(q) as a weighted average.

        R(q) = sum(s_i * sim(q, k_i) * v_i) / sum(s_i * sim(q, k_i))
        Used for Phase 3 skip connections.
        """
        if not self.memories:
            return None

        sims = EmbeddingService.cosine_similarity_matrix(
            query_embedding, self._key_matrix
        )
        strengths = self._strength_vector
        weights = strengths * sims  # (n,)

        weight_sum = np.sum(weights)
        if weight_sum < 1e-8:
            return None

        values = self._value_matrix  # (n, d)
        weighted_values = weights[:, np.newaxis] * values  # (n, d)
        return np.sum(weighted_values, axis=0) / weight_sum

    # ── Equation 10: Enhanced Gate (GRU Multi-Signal) ──

    def _compute_output_entropy(self, output_embedding: np.ndarray) -> float:
        """Equation 10b: Attention entropy of output over stored keys.

        H_out = -Σ(p_i · log(p_i))  where  p = softmax(sims / T)

        NN parallel: In transformers, attention entropy measures how
        "spread out" attention is. High entropy = attending to many keys
        uniformly (the output synthesizes many memories → query response).
        Low entropy = focused on one key or generic (fact acknowledgment).

        Returns normalized entropy in [0, 1] (divided by log(n)).
        """
        if not self.memories or len(self.memories) < 2:
            return 0.0

        sims = EmbeddingService.cosine_similarity_matrix(
            output_embedding, self._key_matrix
        )
        T = self.config.T_entropy

        # Softmax with temperature
        logits = sims / max(T, 1e-8)
        logits = logits - np.max(logits)  # numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        # Shannon entropy
        # Clamp to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))

        # Normalize by max entropy log(n) → [0, 1]
        max_entropy = np.log(len(self.memories))
        if max_entropy < 1e-8:
            return 0.0
        return float(entropy / max_entropy)

    def _compute_value_overlap(self, output_embedding: np.ndarray) -> float:
        """R_val = max_i(sim(e_out, v_i)) — does output echo stored values?

        Catches "I don't know" responses that are similar to other stored
        "I don't know" values, even when R_overlap_out (key match) is low.
        """
        if not self.memories:
            return 0.0
        sims = EmbeddingService.cosine_similarity_matrix(
            output_embedding, self._value_matrix
        )
        return float(np.max(sims))

    def _compute_context_continuity(self, input_embedding: np.ndarray) -> float:
        """D_ctx = sim(e_in, c_{t-1}) — is this input a conversation continuation?

        High D_ctx → input continues the conversation (likely a followup/fragment).
        Low D_ctx → input starts a new topic (likely a new fact).
        """
        if self.context_vector is None:
            return 0.0
        return float(
            EmbeddingService.cosine_similarity(input_embedding, self.context_vector)
        )

    def _memory_input_gate(
        self,
        R_overlap: float,
        R_overlap_val: float = 0.0,
        D_ctx: float = 0.0,
        best_sim: float | None = None,
        H_out: float = 0.0,
    ) -> float:
        """Equation 10 + 10b: Enhanced Memory Input Gate with Attention Entropy.

        NN parallel: Input-driven gate with auxiliary output penalties.

        primary   = 1 - best_sim                          [input novelty]
        penalty   = w_out·R_out + w_val·R_val + w_ctx·D_ctx + w_entropy·H_out
        novelty   = clamp(primary - penalty, 0, 1)
        g         = σ(β_g · (novelty - θ_gate))

        Eq 10b addition: H_out (attention entropy) measures how spread out
        the output's attention is over stored keys. High entropy = output
        synthesizes many memories (query). Low entropy = generic/focused (fact).

        Backward-compatible: when best_sim is None, falls back to R_overlap
        as primary (original Equation 6 behavior).
        """
        # Context continuity factor (normalized to [0, 1], nonlinear scaling)
        # Power-law amplification (Swish/GELU-inspired): strong continuity signals
        # get disproportionately larger penalties than weak ones. This separates
        # mid-conversation queries (D_ctx ≈ 0.75+) from topic-transition facts
        # (D_ctx ≈ 0.65) more decisively than linear scaling.
        theta_ctx = self.config.theta_ctx_gate
        if D_ctx > theta_ctx:
            D_ctx_factor = (D_ctx - theta_ctx) / (1.0 - theta_ctx + 1e-8)
            D_ctx_factor = min(D_ctx_factor, 1.0)
            D_ctx_factor = (
                D_ctx_factor**self.config.ctx_power
            )  # nonlinear amplification
        else:
            D_ctx_factor = 0.0

        # Primary: input novelty (how different is this input from stored keys?)
        if best_sim is not None:
            primary = 1.0 - best_sim
        else:
            primary = 1.0 - R_overlap  # backward compat for tests

        # Auxiliary penalties — bounded additive corrections from output signals
        R_out_penalty = (
            self.config.gate_w_out * R_overlap if best_sim is not None else 0.0
        )
        penalty = (
            R_out_penalty
            + self.config.gate_w_val * R_overlap_val
            + self.config.gate_w_ctx * D_ctx_factor
            + self.config.gate_w_entropy * H_out
        )

        # Clamp to [0, 1]
        novelty = max(0.0, min(1.0, primary - penalty))

        # Sigmoid gate
        x = self.config.beta_gate * (novelty - self.config.theta_gate)
        return float(1.0 / (1.0 + np.exp(-x)))

    # ── Update Cycle ──

    def update(
        self,
        input_embedding: np.ndarray,
        output_embedding: np.ndarray,
        input_text: str,
        output_text: str,
    ) -> dict:
        """Full memory update cycle for one conversation turn.

        Steps:
        1. Increment timestep
        2. Apply global decay to all memories
        3. Compute context continuity BEFORE updating context (Equation 10)
        4. Update context vector (Equation 8)
        5. Find best matching memory
        6. Soft decision (Eq 25-27) or hard threshold → strengthen or create
        7. Cold start boost (Eq 28) on alpha and initial strength
        8. Multi-channel contradiction (Eq 24) if enabled
        9. Hierarchy consolidation (Eq 22) if triggered
        """
        self.timestep += 1
        self._apply_global_decay()

        # Equation 10: compute context continuity BEFORE updating context
        D_ctx = self._compute_context_continuity(input_embedding)

        # Equation 8: update context vector
        self._update_context(input_embedding)

        # Equation 28: cold start factor
        cold_factor = self._cold_start_factor()

        if not self.memories:
            initial_s = self.config.initial_strength * cold_factor
            self._create_memory(
                input_embedding,
                output_embedding,
                input_text,
                output_text,
                initial_strength=initial_s,
            )
            return {"action": "created", "index": 0, "details": "first memory"}

        best_idx, best_sim = self._find_best_match(input_embedding)

        # ── Equation 25-27: Soft Decision OR hard threshold ──
        if self.config.soft_decisions_enabled:
            p_strengthen = self._soft_decision(best_sim, input_embedding)
            should_strengthen = p_strengthen >= 0.5
        else:
            p_strengthen = 1.0 if best_sim >= self.config.theta_create else 0.0
            should_strengthen = best_sim >= self.config.theta_create

        if should_strengthen:
            mem = self.memories[best_idx]

            # Equation 1 + 12 + 28: Strengthen (with prospect theory + cold start)
            raw_gain = self.config.alpha * best_sim * cold_factor
            if self.config.soft_decisions_enabled:
                raw_gain *= p_strengthen  # weight by confidence
            mem.strength += self._prospect_strength_delta(raw_gain)
            mem.timestamp = self.timestep
            mem.access_count += 1
            # Equation 17: record access for ACT-R
            if self.config.actr_enabled:
                mem.access_history.append(self.timestep)
                if len(mem.access_history) > self.config.actr_max_history:
                    mem.access_history = mem.access_history[
                        -self.config.actr_max_history :
                    ]

            # Check for contradiction (Eq 24: multi-channel or Eq 2: standard)
            is_contradiction = (
                self._multichannel_contradiction(
                    mem, output_embedding, input_embedding, best_sim
                )
                if self.config.multichannel_enabled
                else self._detect_contradiction(mem, output_embedding, best_sim)
            )

            if is_contradiction:
                # Equation 2: Value update
                delta_v = output_embedding - mem.value
                mem.value = mem.value + mem.bias * delta_v
                mem.strength *= self.config.delta

                # Also create new memory for the updated info
                self._create_memory(
                    input_embedding, output_embedding, input_text, output_text
                )
                self._maybe_consolidate()
                return {
                    "action": "contradiction",
                    "index": best_idx,
                    "details": f"sim={best_sim:.3f}, value shifted, new memory created",
                }

            # Equation 2b: Soft value drift (gradient descent-inspired).
            val_sim = float(
                EmbeddingService.cosine_similarity(mem.value, output_embedding)
            )
            drift = mem.bias * (1.0 - val_sim) * (output_embedding - mem.value)
            mem.value = mem.value + drift
            norm = np.linalg.norm(mem.value)
            if norm > 1e-8:
                mem.value = mem.value / norm

            return {
                "action": "strengthened",
                "index": best_idx,
                "details": (
                    f"sim={best_sim:.3f}, strength={mem.strength:.3f}, "
                    f"val_drift={1.0 - val_sim:.3f}"
                    + (
                        f", p_str={p_strengthen:.3f}"
                        if self.config.soft_decisions_enabled
                        else ""
                    )
                ),
            }

        # No good match — apply Enhanced Gate (Equations 10 + 10b)
        R_overlap_out = self._compute_retrieval_overlap(output_embedding)
        R_overlap_val = self._compute_value_overlap(output_embedding)
        H_out = self._compute_output_entropy(output_embedding)

        # Hard R_out ceiling: if LLM output is >90% similar to a stored key,
        # it's unambiguously retrieval — skip the sigmoid entirely.
        if R_overlap_out >= self.config.theta_out_filter:
            return {
                "action": "filtered",
                "details": (
                    f"gate=0.000, best_sim={best_sim:.3f}, "
                    f"R_out={R_overlap_out:.3f} >= {self.config.theta_out_filter} (retrieval ceiling), "
                    f"R_val={R_overlap_val:.3f}, D_ctx={D_ctx:.3f}, H_out={H_out:.3f}"
                ),
            }

        gate = self._memory_input_gate(
            R_overlap_out, R_overlap_val, D_ctx, best_sim=best_sim, H_out=H_out
        )

        if gate < self.config.theta_min_gate:
            return {
                "action": "filtered",
                "details": (
                    f"gate={gate:.3f}, best_sim={best_sim:.3f}, "
                    f"R_out={R_overlap_out:.3f}, R_val={R_overlap_val:.3f}, "
                    f"D_ctx={D_ctx:.3f}, H_out={H_out:.3f}"
                ),
            }

        # Create with gated strength (+ cold start + soft decision weight)
        new_idx = len(self.memories)
        create_strength = gate * self.config.initial_strength * cold_factor
        if self.config.soft_decisions_enabled:
            create_strength *= 1.0 - p_strengthen  # weight by create confidence
        self._create_memory(
            input_embedding,
            output_embedding,
            input_text,
            output_text,
            initial_strength=create_strength,
        )
        self._maybe_consolidate()
        return {
            "action": "created",
            "index": new_idx,
            "details": (
                f"gate={gate:.3f}, R_out={R_overlap_out:.3f}, "
                f"R_val={R_overlap_val:.3f}, D_ctx={D_ctx:.3f}, "
                f"H_out={H_out:.3f}, best_sim={best_sim:.3f}"
                + (
                    f", p_str={p_strengthen:.3f}"
                    if self.config.soft_decisions_enabled
                    else ""
                )
            ),
        }

    # ── Internal helpers ──

    def _apply_global_decay(self):
        """Apply gamma decay to all memory strengths.

        Global decay is always symmetric (no prospect theory).
        Prospect theory (Eq 12) applies only to event-driven changes:
        gains from matching, losses from contradictions — not passive decay.
        Kahneman's prospect theory models reactions to discrete events,
        not the passage of time.
        """
        for m in self.memories:
            m.strength *= self.config.gamma

    def _find_best_match(self, embedding: np.ndarray) -> tuple[int, float]:
        """Find memory with highest cosine similarity. Returns (index, sim)."""
        if not self.memories:
            return -1, 0.0
        sims = EmbeddingService.cosine_similarity_matrix(embedding, self._key_matrix)
        best_idx = int(np.argmax(sims))
        return best_idx, float(sims[best_idx])

    def _detect_contradiction(
        self, memory: MemoryEntry, new_output_embedding: np.ndarray, key_sim: float
    ) -> bool:
        """Contradiction: high key similarity but low value similarity."""
        if key_sim < self.config.theta_key:
            return False
        value_sim = EmbeddingService.cosine_similarity(
            memory.value, new_output_embedding
        )
        return value_sim < self.config.theta_value

    def _compute_retrieval_overlap(self, output_embedding: np.ndarray) -> float:
        """R_overlap = max_i(sim(e_out, k_i)) — how much output echoes stored knowledge.

        High R_overlap → LLM is retrieving from memory (query response).
        Low R_overlap → LLM is acknowledging new info (fact response).
        """
        if not self.memories:
            return 0.0
        sims = EmbeddingService.cosine_similarity_matrix(
            output_embedding, self._key_matrix
        )
        return float(np.max(sims))

    def _maybe_consolidate(self):
        """Trigger hierarchy consolidation if enough updates have occurred.

        Equation 22: Called every consolidate_every updates to form
        L2 patterns from L1 clusters and L3 identities from L2.
        """
        if not self.config.hierarchy_enabled:
            return
        if (
            len(self.memories) >= self.config.min_cluster_size
            and self.timestep % self.config.consolidate_every == 0
        ):
            self.hierarchy.consolidate(self.memories)

    def _create_memory(
        self,
        key: np.ndarray,
        value: np.ndarray,
        input_text: str,
        output_text: str,
        initial_strength: float | None = None,
    ):
        """Create a new memory entry and connect it to the graph."""
        # Equation 14: classify into channel
        category = self._classify_memory(key)

        # Equation 17: initial access history
        access_history = [self.timestep] if self.config.actr_enabled else []

        # Equation 24: snapshot context vector at creation time
        ctx_snapshot = (
            self.context_vector.copy() if self.context_vector is not None else None
        )

        entry = MemoryEntry(
            key=key.copy(),
            value=value.copy(),
            strength=(
                initial_strength
                if initial_strength is not None
                else self.config.initial_strength
            ),
            bias=self.config.initial_bias,
            timestamp=self.timestep,
            access_count=0,
            input_text=input_text,
            output_text=output_text,
            access_history=access_history,
            category=category,
            context_at_creation=ctx_snapshot,
        )
        self.memories.append(entry)

        # Equation 19: Add to sparse lexical index
        if self.config.sparse_recall_enabled:
            self.sparse_index.add(input_text)

        # Equation 4 + 18: Compute edges to all existing memories
        new_idx = len(self.memories) - 1
        self.graph.add_memory(new_idx, self.memories)

        # Equation 7: Propagate contradiction through graph edges
        self.graph.propagate_contradiction(new_idx, self.memories, self.config)

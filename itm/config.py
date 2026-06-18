from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """All hyperparameters for the memory layer system."""

    # Embedding (local: BAAI/bge-m3 via sentence-transformers)
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # Strength dynamics (Equation 1)
    alpha: float = 0.1  # learning rate for strength update
    gamma: float = 0.995  # decay factor per timestep
    initial_strength: float = 1.0
    initial_bias: float = 0.5

    # Contradiction detection (Equation 2)
    theta_key: float = 0.7  # key similarity threshold for match
    theta_value: float = 0.5  # value dissimilarity threshold
    delta: float = 0.5  # strength reduction multiplier on contradiction

    # Create vs strengthen (Phase 1: hard threshold, Phase 4: replaced by sigmoid)
    theta_create: float = 0.75

    # Recall (Equation 3)
    top_k: int = 5
    min_relevance: float = 0.01

    # Memory Graph — Equations 4 & 5 (GNN-inspired spreading activation)
    alpha_k: float = 0.5  # key similarity weight in edge computation
    alpha_v: float = 0.5  # value similarity weight in edge computation
    tau_temporal: float = 5.0  # temporal decay scale (turns)
    theta_edge: float = 0.1  # minimum edge weight to form connection
    eta_propagation: float = 0.5  # spreading activation rate
    n_hops: int = 2  # propagation depth (GNN layers)
    activation_threshold: float = 0.01  # minimum activation to include in results

    # Memory Input Gate — Equation 6 (LSTM input gate inspired)
    beta_gate: float = 6.0  # gate sigmoid sharpness (softer for BGE-M3)
    theta_gate: float = 0.4  # gate threshold center (lower = more permissive)
    theta_min_gate: float = 0.12  # hard floor — below this, always skip

    # Graph-Propagated Contradiction — Equation 7 (GNN backpropagation)
    delta_graph: float = 0.7  # strength reduction for graph contradictions

    # Conversation Context Vector — Equation 8 (RNN hidden state)
    lambda_context: float = 0.7  # context persistence (EMA decay)
    beta_context: float = 8.0  # blending sigmoid sharpness
    theta_context: float = 0.3  # self-sufficiency threshold for blend

    # Multi-Head Recall — Equation 9 (multi-head attention)
    recall_w_semantic: float = 0.45  # head 1: sim(q_eff, key)
    recall_w_answer: float = 0.20  # head 2: sim(q_eff, value)
    recall_w_context: float = 0.20  # head 3: sim(c_t, key)
    recall_w_recency: float = 0.15  # head 4: temporal freshness
    tau_recall: float = 10.0  # recency decay scale (timesteps)
    recall_strength_exp: float = (
        0.3  # strength compression exponent (like BM25 TF saturation)
    )

    # Enhanced Gate — Equation 10 (input-driven with output penalties)
    gate_w_out: float = 0.15  # R_out penalty weight (reduced — noisy in BGE-M3)
    gate_w_val: float = (
        0.05  # R_val penalty weight (low: ack responses all similar in BGE-M3)
    )
    gate_w_ctx: float = 0.50  # D_ctx penalty weight (primary discriminator for queries)
    ctx_power: float = (
        1.5  # D_ctx nonlinear scaling (Swish-inspired: amplify strong signals)
    )
    # Attention Entropy Gate — Equation 10b (transformer attention entropy)
    # DISABLED: BGE-M3's compressed similarity space (0.4-0.7 for all pairs) produces
    # near-uniform softmax distributions, so H_out ≈ 1.0 for both facts AND queries.
    # Entropy is non-discriminative — same issue as adaptive heads (Equation 11).
    # Re-enable with wider-range embedding model where query H >> fact H.
    gate_w_entropy: float = (
        0.0  # Eq 10b: attention entropy penalty (disabled for BGE-M3)
    )
    T_entropy: float = 0.5  # Eq 10b: softmax temperature for entropy computation
    theta_ctx_gate: float = 0.5  # context continuity threshold (raised for BGE-M3)
    theta_out_filter: float = (
        0.90  # hard R_out ceiling — above this, always filter (clear retrieval)
    )

    # Adaptive Head Weighting — Equation 11 (self-attention over heads)
    # DISABLED: recency head always has max_score=1.0 while semantic maxes at
    # 0.5-0.8 with BGE-M3, so softmax systematically overweights recency.
    # Re-enable after normalizing head max scores to comparable ranges.
    adaptive_heads: bool = False  # use fixed weights until head normalization
    head_temperature: float = 0.5  # softmax temperature (unused when disabled)

    # Persistence
    memory_dir: str = "memory_data"
    user_id: str = "default_user"

    # ── Phase 2: Equations 12-17 (Kahneman + ACT-R + MemWire) ──

    # Equation 12: Prospect-Theory Asymmetric Strength (Kahneman 1979)
    # NN parallel: Asymmetric loss functions (focal loss, Huber loss).
    # Losses (contradictions/decay) hit harder than gains (reinforcements).
    # v(x) = x^ρ for gains, -λ·(-x)^ρ for losses
    prospect_enabled: bool = False
    prospect_rho: float = 0.88  # diminishing returns exponent (Kahneman: α=β=0.88)
    prospect_lambda_loss: float = 2.25  # loss amplification (Kahneman: λ=2.25)

    # Equation 13: Tension Detection in Recall (WYSIATI-breaking)
    # NN parallel: Contrastive learning / adversarial training.
    # Surface both supporting and conflicting evidence to break the
    # "What You See Is All There Is" trap.
    tension_enabled: bool = False
    theta_tension_key: float = 0.5  # min key sim to compare for tension
    theta_tension_val: float = 0.4  # max value sim to flag as conflict

    # Equation 14: Memory Category Channels (Attribute Substitution fix)
    # NN parallel: Mixture of Experts routing / multi-head channel separation.
    # Different memory types (fact, preference, instruction) routed to
    # different parts of the prompt to prevent preference-biased reasoning.
    channels_enabled: bool = False
    channel_anchors: dict = None  # set in __post_init__

    # Equation 15: Adaptive Recall Depth (Kahneman Capacity Model)
    # NN parallel: Adaptive computation / early exit in transformers.
    # Easy queries need few memories, hard queries need more.
    adaptive_k_enabled: bool = False
    beta_difficulty: float = 1.0  # difficulty scaling factor
    top_k_max: int = 15  # hard upper bound on adaptive k

    # Equation 16: Feedback Loop (Post-Response Edge Strengthening)
    # NN parallel: RLHF reward signal / Hebbian post-hoc learning.
    # After LLM responds, strengthen paths it used, weaken paths it ignored.
    feedback_enabled: bool = False
    theta_align: float = 0.6  # alignment threshold for strengthen
    theta_misalign: float = 0.3  # misalignment threshold for weaken
    alpha_feedback: float = 0.05  # feedback learning rate (strength)
    eta_feedback: float = 0.1  # feedback edge adjustment rate

    # Equation 17: ACT-R Base-Level Activation (Power-Law Decay)
    # NN parallel: ACT-R cognitive architecture (Anderson 1976-2025).
    # B_i = ln(Σ_n (t - t_n)^(-d)) — power-law based on access history.
    # Validated against 50 years of human memory experiments.
    actr_enabled: bool = False
    actr_decay: float = 0.5  # power-law decay exponent (ACT-R: d=0.5)
    actr_max_history: int = 20  # max access timestamps to store

    # ── Phase 3: Equations 18-21 (MemWire-Inspired, Formalized) ──

    # Equation 18: Displacement-Augmented Edges (ResNet residual connections)
    # d_i = normalize(v_i - k_i) captures relationship type (person→location, etc.)
    # Similar displacements = similar relationship types → stronger edges.
    displacement_edges_enabled: bool = False
    alpha_d: float = 0.3  # displacement similarity weight in edge computation
    # When enabled, alpha_k and alpha_v are scaled: total = alpha_k + alpha_v + alpha_d

    # Equation 19: Sparse Lexical Recall (BM25, Robertson 1994)
    # Dense embeddings miss exact keyword matches (phone numbers, names, codes).
    # TF-IDF sparse vectors catch what dense misses.
    sparse_recall_enabled: bool = False
    recall_w_sparse: float = 0.15  # head 5: sparse lexical similarity weight
    sparse_min_df: int = 1  # min document frequency for IDF

    # Equation 20: Multi-Scale Activation (JK-Net, Xu et al. 2018)
    # Combine activations from ALL depths, not just final hop.
    # 1-hop = local context, 4-hop = global reach.
    multiscale_enabled: bool = False
    multiscale_hops: int = 4  # number of propagation hops (was n_hops=2)
    multiscale_damping: float = 0.7  # per-hop damping: η_l = η · damping^l
    # Scale weights: uniform by default (each hop contributes equally)

    # Equation 21: Query-Aware Message Passing (GAT, Veličković 2018)
    # Messages weighted by sender's relevance to original query.
    # Prevents irrelevant but well-connected memories from stealing activation.
    query_aware_spread_enabled: bool = False

    # ── Phase 4: Equations 22-29 (Research.md §14-18) ──

    # Equation 22-23: Hierarchical Memory L1→L2→L3 (MLP 1986 + CNN Pooling 1998)
    # NN parallel: MLP hidden layers compress representations into abstractions.
    # CNN pooling reduces spatial dimensions while preserving key features.
    # ResNet skip connections (He 2015) prevent info loss in deep hierarchies.
    #
    # L2 Formation: K_L2 = Σ(s_i·k_i)/Σ(s_i), S_L2 = Σ(s_i)
    # L3 Formation: same pooling over L2 patterns
    # Recall: R(q) = w₁·R_L1 + w₂·R_L2 + w₃·R_L3  (skip connections)
    hierarchy_enabled: bool = False
    theta_cluster: float = 0.7  # min avg pairwise sim for L2 formation
    min_cluster_size: int = 3  # min L1 memories to form L2
    theta_identity: float = 0.6  # min sim for L3 formation
    min_identity_size: int = 2  # min L2 patterns to form L3
    recall_w1: float = 0.5  # L1 skip connection weight
    recall_w2: float = 0.3  # L2 weight
    recall_w3: float = 0.2  # L3 weight
    consolidate_every: int = 20  # auto-consolidate interval

    # Equation 24: Multi-Channel Similarity (Multi-Head Attention, Vaswani 2017)
    # NN parallel: Different attention heads capture different aspects of meaning.
    # Topic channel = key embedding, Displacement channel = normalize(v-k) from Eq 18,
    # Context channel = context vector at memory creation time (Eq 8).
    #
    # sim_mc(q, m_i) = w_t·sim(q, k_i) + w_d·sim(d_q, d_i) + w_c·sim(c_q, c_i)
    #
    # For contradiction: high topic sim + low displacement sim → opposing stances.
    # No LLM calls — all channels derived from existing embeddings.
    multichannel_enabled: bool = False
    w_channel_topic: float = 0.5  # topic channel weight
    w_channel_disp: float = 0.3  # displacement/stance channel weight
    w_channel_ctx: float = 0.2  # context channel weight

    # Equations 25-27: Soft Create-vs-Strengthen (Sigmoid Activation, 1986)
    # NN parallel: Sigmoid replacing step function enabled backpropagation —
    # the single most important transition in ML history.
    #
    # Eq 25: p_strengthen = σ(β·(sim_best - θ_local))
    #         p_create = 1 - p_strengthen
    #         Both ops happen with their respective probabilities.
    #
    # Eq 26: Adaptive Sharpness (Temperature Scaling)
    #         β = β_min + (β_max - β_min) · min(n/n_mature, 1)
    #         Few memories → soft boundary (explore). Many → sharp (exploit).
    #
    # Eq 27: Per-Region Adaptive Threshold (Adam/BatchNorm)
    #         θ_local = θ_base + β_θ · density(q)
    #         Dense regions → high threshold. Sparse → low.
    soft_decisions_enabled: bool = False
    beta_soft_min: float = 5.0  # min sharpness (early, exploring)
    beta_soft_max: float = 20.0  # max sharpness (mature, decisive)
    n_mature: int = 100  # memories until max sharpness
    theta_soft_base: float = 0.55  # base threshold for soft decision
    beta_theta_density: float = 0.3  # density scaling for adaptive threshold
    theta_density_neighbor: float = 0.5  # sim threshold for density counting

    # Equation 28: Cold Start Learning Rate Boost (Inverted Warmup)
    # NN parallel: Learning rate warmup (Goyal 2017) but inverted.
    # Standard warmup: start low → ramp up. Ours: start HIGH → decay to normal.
    # First memories need aggressive learning to establish patterns quickly.
    #
    # α_eff = α · (1 + κ · exp(-n_memories / τ_cold))
    # s_initial_cold = s_initial · (1 + κ/2 · exp(-n_memories / τ_cold))
    cold_start_enabled: bool = False
    cold_start_kappa: float = 3.0  # max boost multiplier
    cold_start_tau: float = 10.0  # decay rate (memories until baseline)

    # Equation 29: Shared Memory Layer (Federated Averaging, McMahan 2017)
    # NN parallel: LoRA adapters (Hu 2021) + Federated Learning.
    # Personal memory = lightweight adapter. Shared = team knowledge base.
    #
    # R(q) = (1 - w_shared) · R(q, M_personal) + w_shared · R(q, M_shared)
    shared_memory_enabled: bool = False
    shared_memory_dir: str = ""  # path to shared memory directory
    shared_memory_weight: float = 0.3  # weight for shared memory in recall

    def __post_init__(self):
        if self.channel_anchors is None:
            self.channel_anchors = {
                "fact": "This is a factual statement about the world or about me",
                "preference": "I like this, I prefer this, my favorite is",
                "instruction": "Please always do this, remember to do this",
                "event": "Something happened, I went to, yesterday I did",
            }

# Inference-Time Memory (ITM)

A long-term memory layer for LLM agents whose memory operations are pure
numpy on a local embedder. Recall, strength updates, contradiction handling,
graph spreading, and consolidation are defined by 29 neural-network-derived
equations and run with zero LLM API calls. The only model ITM loads for memory
work is a local sentence embedder (BGE-M3 by default).

[![Tests](https://img.shields.io/badge/Tests-197_passing-brightgreen.svg)]()
[![Equations](https://img.shields.io/badge/Equations-29-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Large Language Models are frozen after training. They cannot accumulate facts
across a long conversation or revise an earlier statement. ITM is a small
memory substrate that sits next to a frozen model: it ingests dialogue,
strengthens what gets used, lets corrections drift the stored representation,
and retrieves evidence at query time. Every one of those operations is a
closed-form update over embedding vectors, derived from a specific neural
network mechanism (Perceptron, LSTM gate, GNN message passing, GAT attention,
BM25, ACT-R decay, prospect theory).

What ITM does NOT do: it does not call an LLM to classify, extract, or
summarize memories, and it does not improve the answering model's reasoning.
Its job is to put the right evidence in front of the model. The measured
results below show that this is where most of the lift comes from, and also
where the ceiling is.

---

## What is true and measured (read this first)

ITM was evaluated on the full LOCOMO benchmark (10 conversations, 1986
questions) with `gpt-4.1-nano` answering and judging, BGE-M3 embeddings, and
`top_k=5`. The headline is a controlled ablation of two memory behaviors, not
an absolute-accuracy or head-to-head claim:

- On answerable questions (LOCOMO cats 1-4, n=1540), judge accuracy goes
  3.57% -> 19.74% -> 30.58% as two flags are turned on (8.6x relative lift).
- The reason is retrieval: gold evidence appears in the recalled top-5
  5.3% -> 41.3% -> 65.6% of the time.
- The honest ceiling: even when gold IS in context, `gpt-4.1-nano` answers
  correctly only ~38% of the time. ITM retrieves the evidence; the answer
  model is the bottleneck.

The full table, the metrics, the robustness checks, and the caveats are in
[Results](#results). The benchmark is reproducible from this repo; see
[`benchmarks/README.md`](benchmarks/README.md).

---

## Design properties

These are the genuine, code-backed properties of the system.

- Zero LLM API calls for memory operations. Recall, the strength/value
  updates, contradiction drift, graph spreading, and consolidation are numpy
  over embedding vectors (`itm/core.py`, `itm/graph.py`, `itm/hierarchy.py`).
  Embeddings are computed locally by default (BGE-M3 via
  `sentence-transformers`); the optional `[api]` backends are paid services.
- Continuous strength learning. Each memory has a learned strength updated by
  a Perceptron-style rule (Eq 1) with ACT-R power-law decay (Eq 17). Nothing
  is pruned; strength encodes relevance, not existence.
- Contradiction by drift, not delete. A correction moves the stored value
  vector toward the new statement (Eq 2) instead of removing and re-inserting.
- Graph recall. Memories are linked by semantic/temporal proximity (Eq 4) and
  activation spreads through the graph (Eq 5), with GAT-style query-aware
  message passing (Eq 21) and JK-Net multi-scale pooling (Eq 20).
- Hybrid retrieval. A multi-head recall (semantic, answer, context, recency)
  combines with a BM25 sparse lexical head (Eq 9, 19) so exact-keyword matches
  are not lost in dense space.
- 197 unit tests, including per-equation property tests, on a deterministic
  fake embedder. Lint (ruff), format (black), type-check (mypy), and the test
  suite run in GitHub Actions CI.

### Latency, honestly

Recall is synchronous: ITM embeds the query and runs the numpy recall before
the LLM call, so it adds latency to each request. Only the post-response
memory update (embed the output, update strengths/edges, persist) runs in a
background thread and does not block the response. This is described in the
`itm/patch.py` module docstring and is the actual behavior of `enable_memory`.

---

## The 29 Equations

Every memory behavior is a closed-form update mapped to a neural-network
mechanism. No heuristics, no LLM classification.

### Phase 1: Core Memory (Perceptron + GNN)

| # | Name | Equation | Neural Network Origin |
|---|------|----------|----------------------|
| 1 | Strength Update | `s_i(t+1) = s_i·γ + α·sim(e_in, k_i)` | Perceptron weight update (1958) |
| 2 | Value Drift (hard) | `v_i(t+1) = normalize(v_i + b_i·(e_out_new - v_i))` | Gradient descent / bias adaptation |
| 2b | Value Drift (soft) | `v_i(t+1) = normalize(v_i + b_i·(1 - sim(v_i,e_out))·(e_out - v_i))` | Soft-drift counterpart to Eq 2 (similarity-gated) |
| 3 | Recall | `R(q) = Σ(s_i·sim(q,k_i)·v_i) / Σ(s_i·sim)` | Weighted retrieval |
| 4 | Association Edge | `w_ij = (α_k·sim_k + α_v·sim_v)·exp(-\|Δt\|/τ)` | Hebbian learning (1949) |
| 5 | Spreading Activation | `a_i^(l+1) = a_i^(l) + η·Σ_j(w_ij·a_j^(l))` | GNN message passing (2017) |
| 6 | Memory Input Gate | `g = σ(β_g·((1-R_overlap) - θ_gate))` | LSTM input gate (1997) |
| 7 | Graph Contradiction | `v_j += b_j·(v_i - v_j), s_j *= δ` | Backpropagation through graph |

### Phase 1.5: Advanced Recall (RNN + Multi-Head Attention)

| # | Name | Equation | Neural Network Origin |
|---|------|----------|----------------------|
| 8 | Context Vector | `c_t = λ·c_{t-1} + (1-λ)·e_in_t` | RNN hidden state |
| 9 | Multi-Head Recall | semantic + answer + context + recency heads | Multi-head attention (2017) |
| 10 | Enhanced Gate | Primary + R_out/R_val/D_ctx penalties + hard ceiling | Deep gate networks |
| 11 | Adaptive Head Weights | Softmax over head confidences (default off) | Mixture of experts |

### Phase 2: Cognitive Science (Kahneman + ACT-R)

| # | Name | Equation | Origin |
|---|------|----------|--------|
| 12 | Prospect Strength | `Δs_gain·1.0` vs `Δs_loss·λ` (λ=2.25) | Kahneman prospect theory (1979) |
| 13 | Tension Detection | conflict among recalled memories | WYSIATI-breaking conflict detection |
| 14 | Category Channels | fact / preference / instruction / event routing | Dual-process theory |
| 15 | Adaptive-k Recall | k scales with query difficulty | ACT-R retrieval threshold |
| 16 | Feedback Loop | Strengthen/weaken recalled based on response | Reinforcement learning |
| 17 | ACT-R Decay | `s_i *= (1 + t_idle)^(-d)` | ACT-R power-law decay (Anderson 1993) |

### Phase 3: Retrieval (ResNet + BM25 + JK-Net + GAT)

| # | Name | Equation | Neural Network Origin |
|---|------|----------|----------------------|
| 18 | Displacement Edges | `d_i = normalize(v_i - k_i)`; renormalized into Eq 4 (convex) | ResNet residual connections (2015) |
| 19 | Sparse Lexical Recall | TF-IDF sparse vectors + BM25 scoring | BM25 (Robertson 1994) |
| 20 | Multi-Scale Activation | `a_final = Σ_l(w_l·a^(l))` with per-hop damping | JK-Net (Xu et al. 2018) |
| 21 | Query-Aware Spreading | `msg_ij = w_ij·a_j·sim(q, k_j)` | GAT attention (Velickovic 2018) |

### Phase 4: Scaling & Robustness

| # | Name | Equation | Neural Network Origin |
|---|------|----------|----------------------|
| 22 | L2 Pattern Formation | `K_L2 = Σ(s_i·k_i)/Σ(s_i)`, `S_L2 = Σ(s_i)` | CNN weighted pooling (LeCun 1998) |
| 23 | Hierarchical Recall | `R(q) = w₁·R_L1 + w₂·R_L2 + w₃·R_L3` | ResNet skip connections (He 2015) |
| 24 | Multi-Channel Contradiction | `sim_mc = w_t·topic + w_d·disp + w_c·ctx` | Multi-head attention (Vaswani 2017) |
| 25 | Soft Create-vs-Strengthen | `p_str = σ(β·(sim - θ_local))` | Sigmoid activation (1986) |
| 26 | Adaptive Sharpness | `β = β_min + (β_max-β_min)·saturation` | Temperature scaling (softmax) |
| 27 | Adaptive Threshold | `θ_local = θ_base + β_θ·density(q)` | Adam optimizer / BatchNorm |
| 28 | Cold Start Boost | `α_eff = α·(1 + κ·exp(-n/τ))` | Inverted learning-rate warmup |
| 29 | Shared Memory Merge | `R = (1-w_s)·R_personal + w_s·R_shared` | Federated averaging (McMahan 2017) |

Equation 29 is a configuration placeholder (the merge weight exists in
`config.py`); the full cross-store merge is not implemented.

---

## Quickstart

### Installation

Requires Python 3.12+. Core dependencies (`numpy`, `sentence-transformers`,
`openai`, `python-dotenv`) are declared in `pyproject.toml`.

```bash
# Editable install with runtime dependencies
pip install -e .

# With dev tooling (pytest, ruff, black, mypy)
pip install -e .[dev]

# With optional alternate embedding backends (cohere, nomic)
pip install -e .[api]
```

### Usage

Two lines to add memory to an OpenAI client:

```python
from openai import OpenAI
from itm.config import MemoryConfig
from itm.patch import enable_memory

client = OpenAI()

# Enable the Phase 2/3 equations (Eq 12-21)
config = MemoryConfig(
    prospect_enabled=True,              # Eq 12
    tension_enabled=True,               # Eq 13
    channels_enabled=True,              # Eq 14
    adaptive_k_enabled=True,            # Eq 15
    feedback_enabled=True,              # Eq 16
    actr_enabled=True,                  # Eq 17
    displacement_edges_enabled=True,    # Eq 18
    sparse_recall_enabled=True,         # Eq 19
    multiscale_enabled=True,            # Eq 20
    query_aware_spread_enabled=True,    # Eq 21
)

mem = enable_memory(client, config)

# Recall runs synchronously before the call; the memory update is backgrounded.
response = client.responses.create(
    model="gpt-4.1-nano",
    input="Hi, I'm a new user. I study computer science.",
)
print(response.output_text)

mem.flush()      # wait for background memory updates
mem.shutdown()
```

### Interactive Demo

```bash
python gpt.py

# Commands inside the demo:
stats              # memory statistics
memories           # list all stored memories
recall <query>     # search memories by query
```

---

## Architecture

```
User Input --> LLM (frozen) --> Response
     |                              |
  embed(input)                embed(output)
     |                              |
  +------------------------------------------+
  |        MEMORY LAYER (29 equations)       |
  |                                          |
  |  Phase 1: Core                           |
  |    Eq 1-3: Strength / Value / Recall     |
  |    Eq 4-5: Graph Edges + GNN Spreading   |
  |    Eq 6-7: LSTM Gate + Graph Backprop    |
  |                                          |
  |  Phase 1.5: Advanced Recall              |
  |    Eq 8: RNN Context Vector              |
  |    Eq 9: Multi-Head Recall               |
  |    Eq 10-11: Enhanced Gate + Adaptive    |
  |                                          |
  |  Phase 2: Cognitive Science              |
  |    Eq 12: Prospect Theory Gating         |
  |    Eq 13-14: Tension + Channels          |
  |    Eq 15-17: Adaptive-k + ACT-R Decay    |
  |                                          |
  |  Phase 3: Retrieval                      |
  |    Eq 18: ResNet Displacement Edges      |
  |    Eq 19: BM25 Sparse Lexical Recall     |
  |    Eq 20: JK-Net Multi-Scale Activation  |
  |    Eq 21: GAT Query-Aware Spreading      |
  +------------------------------------------+
```

Recall (the query path) blocks the LLM call. The update path (right side) runs
in a background thread after the response.

## Project Structure

```
itm/
  config.py          - All hyperparameters (29 equations worth)
  core.py            - MemoryEntry + MemoryLayer + SparseIndex
  graph.py           - MemoryGraph (GNN edges + spreading activation)
  hierarchy.py       - L1/L2/L3 consolidation + hierarchical recall
  embeddings.py      - BGE-M3 local embeddings (on-device)
  embeddings_api.py  - Optional alternate backends (Nomic, OpenAI, Cohere; paid)
  storage.py         - Save/load to disk (.npz + .json)
  patch.py           - Transparent OpenAI client adapter
  formatting.py      - Memory display + prompt formatting
  stats.py           - CLI statistics and graph search

tests/               - 197 tests across the suite (offline FakeEmbedder)
benchmarks/          - LOCOMO evaluation harness and results (see its README)
gpt.py               - Interactive chat demo with full memory
test_gpt.py          - Real-BGE-M3 integration smoke test (not run in CI)
```

---

## Results

Full LOCOMO benchmark (ACL 2024), 10 conversations, 1986 questions.
Answer model and judge model: `gpt-4.1-nano`. Embeddings: BGE-M3. `top_k=5`,
`n_hops=2`. Single seed, no error bars. Numbers below are reproduced directly
from the per-question result files in `benchmarks/results/` and consolidated
in `benchmarks/results/ablation_summary.json`.

The ablation toggles two memory behaviors, both default OFF:

- A = `transcript_ingest`. Store each dialogue turn as unit-strength evidence.
  Without it, the chat-tuned input gate filtered the large majority of
  transcript turns, so gold evidence was almost never stored (gold-in-top5
  5.3% at baseline vs 65.6% with both flags; about 588 memories per
  conversation once the flag is on).
- B = `spreading_debias_enabled`. Suppress the non-discriminative
  ingestion-recency activation floor and apply symmetric GCN normalization to
  graph spreading.

### Answerable questions (LOCOMO cats 1-4, n=1540)

| Metric | baseline (A0B0) | +A (A1B0) | +A+B (A1B1) |
|---|---|---|---|
| Judge accuracy | 3.57% | 19.74% | 30.58% |
| Token-F1 | 0.043 | 0.188 | 0.280 |
| Refusal rate | 86.9% | 51.4% | 33.4% |
| Gold evidence in top-5 | 5.3% | 41.3% | 65.6% |
| Judge accuracy when gold IS in context | 33.3% | 34.9% | 37.6% |

### Per-category judge accuracy

| Category | baseline | +A | +A+B |
|---|---|---|---|
| Multi-hop (cat 1) | 4.6% | 10.6% | 20.2% |
| Temporal (cat 2) | 1.9% | 13.7% | 21.5% |
| Open-domain (cat 3) | 1.0% | 1.0% | 10.4% |
| Single-hop (cat 4) | 4.2% | 27.2% | 39.8% |
| Adversarial (cat 5) | 98.0% | 89.7% | 85.7% |

Overall judge accuracy including the adversarial category: 24.8% -> 35.5% ->
43.0%.

### Robustness

- Refusal-hardened re-scoring. If a refusal ("Not mentioned") on an answerable
  question is forced to a wrong score before averaging, the slope holds:
  3.51% -> 19.74% -> 30.58%. (The two numbers are nearly identical because the
  judge already marks a refusal on an answerable question as wrong.)
- Independent judge. A paired 200-question stratified subsample (the two
  conversations present in all three result files) was re-graded with
  `gpt-4.1-mini` on the same stored predictions, no regeneration. The
  improvement direction is preserved (baseline < +A+B). On this small slice the
  mini judge is more lenient on the baseline -- it credits some
  refusal-adjacent predictions -- so its slope is flatter, and nano-vs-mini
  agreement is 0.745 across the judged pairs. See
  `benchmarks/results/rejudge_mini_subsample.json`.

### Honest ceiling and limitations

- The answer model is the bottleneck, not retrieval. Even with gold evidence
  in the top-5 context, `gpt-4.1-nano` answers correctly only ~37.6% of the
  time. ITM's contribution is "get the right evidence in front of the model"
  (gold-in-top5 rose from 5.3% to 65.6%); it does not make the model reason
  better.
- Open-domain (cat 3, 10.4%) is the weakest category, followed by multi-hop
  (cat 1, 20.2%).
- Better recall costs a few points of appropriate abstention on adversarial
  trap questions (cat 5: 98.0% -> 85.7%). Storing more evidence makes the model
  more willing to answer, including when it should refuse.
- `top_k=5` is a lower bound; larger k was not swept.
- Single seed, no error bars. These are relative-improvement / ablation
  results, not SOTA-competitive absolute accuracy.

### Relation to other systems

Published LOCOMO numbers for retrieval/memory systems such as RAG variants,
mem0, MemGPT, and Zep are typically in the 60-75% range, but under different
answer models, prompts, retrieval budgets, and scoring. This repo contains no
head-to-head measurement against those systems, and ITM's absolute accuracy
here (30.58% on answerable) is well below them. ITM's claim is narrow and
internal: an ablation showing that two memory behaviors raise gold-evidence
retrieval and answerable accuracy by a large relative factor, with all memory
operations done in numpy and no LLM calls.

Reproduce everything in [`benchmarks/README.md`](benchmarks/README.md).

---

## Running Tests

```bash
# All unit tests (197 tests, offline, deterministic FakeEmbedder)
pip install -e .[dev]
python -m pytest -q tests

# Integration smoke test with real BGE-M3 embeddings
# (downloads the model, needs network; illustrative, not a benchmark)
python test_gpt.py
```

`test_gpt.py` is an end-to-end smoke test on a small hand-written set of facts
with simulated responses. It exercises the pipeline; it is not a benchmark and
its pass count is not an accuracy claim. The accuracy numbers are in
[Results](#results).

---

## Research

The theoretical foundations span neural network history -- from Rosenblatt's
Perceptron (1958) to Graph Attention Networks (2018) -- with each equation
mapped to its origin in the tables above (ACT-R power-law decay, prospect
theory weighting, BM25 sparse retrieval, GNN/GAT spreading, ResNet residual
edges).

---

## License

[MIT](LICENSE).

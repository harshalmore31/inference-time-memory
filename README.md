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
Its job is to put the right evidence in front of the model.

Read the measured results before the equation tables. The strongest finding in
this repo is a negative one -- an input gate tuned on interactive chat silently
discarded 87-94% of turns when fed pre-recorded transcripts -- and the recall
machinery, measured against trivial retrievers on the same corpus, beats plain
cosine similarity by 2.7 points and ties a five-line dense+BM25 hybrid. The
equation count is a description of the implementation, not a claim about how
much of it is load-bearing.

---

## What is true and measured (read this first)

ITM was evaluated on the full LOCOMO benchmark (10 conversations, 1986
questions) with `gpt-4.1-nano` answering and judging, BGE-M3 embeddings, and
`top_k=5`. Three claims, in descending order of how well they are supported.

**1. The main result is a negative one, about a gate.** ITM's input gate (Eq 10)
was tuned on interactive chat, where a high context-continuity signal marks a
mid-conversation follow-up worth filtering. In a pre-recorded transcript every
turn is a continuation *by construction*, so that signal carries no information
while still consuming threshold headroom, and 87-94% of turns were discarded.
Gold evidence was therefore almost never stored. Turning the gate off
(`transcript_ingest`) and de-biasing recall (`spreading_debias_enabled`) moves
answerable accuracy 3.57% -> 19.74% -> 30.58% and gold-in-top-5 5.3% -> 41.3%
-> 65.6%.

Read that as *a hyperparameter regime that silently failed to transfer*, not as
evidence the memory layer works. `transcript_ingest=True` returns early in
`itm/core.py`, bypassing the gate, the create-vs-strengthen merge, global decay
and cold start, so the winning arm is closer to "store every turn" than to the
full 29-equation pipeline. The ablation measures ITM recovering from its own
miscalibration.

**2. Against trivial retrieval, the recall machinery wins narrowly and is
matched by a hybrid.** With corpus, embeddings, questions and `top_k` held
fixed (n=1536 paired, zero API calls, `benchmarks/retrieval_baseline.py`):

| retriever | recall@5 | paired exact McNemar vs ITM |
|---|---|---|
| ITM (all 29 equations) | 65.8% | - |
| cosine similarity, one line | 63.0% | ITM +2.73pp, p=0.012 |
| dense + BM25 hybrid, ~5 lines | **67.3%** | ITM -1.50pp, p=0.130 |
| BM25 alone | 51.2% | ITM +14.58pp, p<1e-6 |

ITM is significantly better than plain cosine and statistically
indistinguishable from a five-line hybrid. Per category it is *worse* than
cosine on multi-hop (54.3% vs 57.8%) -- the category graph spreading exists for
-- and better on temporal (76.6% vs 70.1%) and open-domain (42.4% vs 38.0%).

**3. Absolute accuracy is capped downstream of retrieval, but the cause is not
yet isolated.** Even when gold evidence is in the top-5, `gpt-4.1-nano` answers
correctly only ~38% of the time. Part of that is the reader; part is that
`itm/formatting.py` truncates each memory to 120 characters while 69.4% of
LOCOMO turns are longer, so 35.7% of retrieved evidence text never reaches the
prompt. ITM's contribution is getting evidence in front of the model, not
improving reasoning -- but see
[Honest ceiling and limitations](#honest-ceiling-and-limitations) before
treating ~38% as a reader ceiling.

A caution that applies to every LOCOMO number, including the ones above: the
benchmark mixes answerable and adversarial questions, and a system that refuses
everything scores ~100% on the adversarial split for free. See
[Abstention operating curve](#abstention-operating-curve) before reading any
single accuracy figure.

The full tables, metrics, robustness checks and caveats are in
[Results](#results). Everything is reproducible from this repo; see
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

Every memory behavior is a closed-form update over embedding vectors, mapped to
a neural-network mechanism, and computed without any LLM call.

An honest caveat on "derived": the *form* of each update is taken from the cited
mechanism, but the constants are not derived from it. Thresholds, head weights
and sharpness parameters were hand-tuned against BGE-M3's similarity
distribution, and several are explicitly disabled because that distribution
makes them non-discriminative (`adaptive_heads`, `gate_w_entropy`). The
measured consequence of tuning against one input regime is documented above.
Treat the table as a map from mechanism to implementation, not as a claim that
the constants fall out of the theory.

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
43.0%. **Do not read the first of those as a baseline capability.** A policy
that answers nothing at all scores 0% on the 1540 answerable questions and
~100% on the 446 adversarial ones, i.e. 446/1986 = **22.5% overall**. The A0B0
baseline's 24.8% sits 2.3 points above that floor, and its 98.0% adversarial
accuracy is bought entirely by refusing 86.9% of answerable questions. This is
why the tables above lead with the answerable split, and why the adversarial
regression in the next section cannot be read as a straight loss.

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

### Abstention operating curve

Refusal and answering are one tradeoff, not two results. Reporting a single
adversarial accuracy is meaningless without saying where on that tradeoff the
system was sitting. This sweeps the retrieval budget -- how much evidence
reaches the reader -- and traces the frontier
(`benchmarks/operating_curve.py`, stratified subsample of 300 answerable + 200
adversarial questions, `gpt-4.1-nano`, A1B1 config, seed 0):

| top_k | answerable acc | adversarial acc | refusal (answerable) | gold@k |
|---|---|---|---|---|
| 0 (empty context) | 7.0% | 90.0% | 57.3% | 0.0% |
| 1 | 20.7% | 92.0% | 50.3% | 36.3% |
| 2 | 27.7% | 91.5% | 40.3% | 47.7% |
| 3 | 28.3% | 88.0% | 37.0% | 56.0% |
| 5 | 29.0% | 86.0% | 29.3% | 66.3% |
| 8 | 32.7% | 82.0% | 25.0% | 75.0% |
| 12 | 32.3% | 73.0% | 23.0% | 79.7% |
| 20 | 35.3% | 66.0% | 22.3% | 83.7% |
| 30 | 34.7% | 66.0% | 21.3% | 89.0% |

Three things this makes visible that the three-arm ablation cannot:

- **The adversarial score is mostly not a capability.** With k=0 -- literally no
  memory at all -- adversarial accuracy is already 90.0%. The published
  baseline's 98.0% lives in that regime. The 98.0% -> 85.7% "regression" is
  movement along this curve, not a loss of a skill the system had.
- **Retrieval stops being the binding constraint around k=8.** From k=8 to
  k=30, gold-in-top-k climbs 75.0% -> 89.0% while answerable accuracy moves
  32.7% -> 34.7%, inside noise at this sample size. Past that point extra
  evidence buys nothing on answerable questions and costs 16 points of
  adversarial accuracy. The published `top_k=5` is on the steep part of the
  curve, not at a tuned optimum.
- **The subsample tracks the full run.** At k=5 this curve gives 29.0%
  answerable / 86.0% adversarial against 30.6% / 85.7% on all 1986 questions,
  which is the consistency check that makes the rest of the curve credible.

Any comparison of abstention behavior between two systems, or between two
configurations of this one, should be made at a matched point on this curve.

### Honest ceiling and limitations

- The answer model is *a* bottleneck: even with gold evidence in the top-5,
  `gpt-4.1-nano` answers correctly only ~37.6% of the time. But that number is
  confounded and should not yet be read as a pure reader-capability ceiling.
  `itm/formatting.py` truncates each recalled memory to 120 characters of input
  text and 100 of output text, while 69.4% of LOCOMO turns are longer than 120
  characters; **35.7% of retrieved evidence text never reaches the model**. A
  turn can be retrieved into the top-5 and still have the answer-bearing span
  cut off, so "gold in context" and "gold in the prompt" are not the same
  event. Raising the truncation limit is an untested lever on the absolute
  numbers, and it must be ruled out before the ceiling is attributed to the
  reader.
- Open-domain (cat 3, 10.4%) is the weakest category, followed by multi-hop
  (cat 1, 20.2%).
- Retrieving more evidence makes the model more willing to answer, including
  when it should refuse: adversarial accuracy falls 98.0% -> 85.7% while
  refusal on answerable questions falls 86.9% -> 33.4%. Both numbers are
  single points on a tradeoff, not independent results -- see
  [Abstention operating curve](#abstention-operating-curve). The baseline's
  98.0% is what an almost-always-refusing system gets for free, so the honest
  comparison is at matched refusal rates, not at these two arbitrary points.
- The recall machinery is not what produces most of the lift. Against
  trivial retrievers on the same corpus it is +2.7pp over plain cosine
  (p=0.012) and statistically tied with a dense+BM25 hybrid (p=0.13), and it
  loses to both on multi-hop. See the table in
  [What is true and measured](#what-is-true-and-measured-read-this-first).
- `top_k=5` is a lower bound; larger k was not swept.
- Single seed, no error bars. These are relative-improvement / ablation
  results, not SOTA-competitive absolute accuracy.

### Relation to other systems

Published LOCOMO numbers for retrieval/memory systems such as RAG variants,
mem0, MemGPT, and Zep are typically in the 60-75% range, but under different
answer models, prompts, retrieval budgets, and scoring. This repo contains no
head-to-head measurement against those systems, and ITM's absolute accuracy
here (30.58% on answerable) is well below them. Nothing here should be read as
a ranking against them.

The one head-to-head this repo *does* support is against trivial retrievers,
because there everything but the retriever is held fixed: dense cosine, BM25,
and a dense+BM25 hybrid over the identical corpus and embeddings
(`benchmarks/retrieval_baseline.py`, table in
[What is true and measured](#what-is-true-and-measured-read-this-first)). ITM
beats plain cosine by 2.7pp and ties the hybrid. That is the honest size of the
recall machinery's contribution, and it is much smaller than the headline
ablation suggests.

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

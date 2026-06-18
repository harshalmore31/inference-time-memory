# Inference-Time Memory (ITM)

**A Neural Memory Layer for Continuous Learning in Frozen Language Models**

**29 equations. Zero LLM calls for memory ops. Pure math.**

[![Tests](https://img.shields.io/badge/Tests-179_passing-brightgreen.svg)]()
[![Equations](https://img.shields.io/badge/Equations-29-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Large Language Models are frozen after training. They cannot learn from interactions, adapt to contradictions, or consolidate facts over time. Current solutions (RAG, mem0, MemGPT) rely on LLM calls for memory classification and extraction — they are storage systems, not learning systems.

**Inference-Time Memory (ITM)** is a differentiable, graph-based cognitive architecture that runs alongside any frozen LLM. Inspired by 70 years of neural network history — from the Perceptron (1958) to Graph Attention Networks (2018) — ITM gives language models a living memory that strengthens, decays, associates, and adapts in real-time using pure mathematical equations.

---

## Why ITM is Different

| | RAG / mem0 / MemGPT | **ITM** |
|---|---|---|
| **Memory ops** | LLM calls for classify/extract/summarize | Zero LLM calls — pure math |
| **Learning** | Static storage + retrieval | Continuous Hebbian learning |
| **Contradictions** | Delete + re-insert | Gradient-descent value drift (Eq 2) |
| **Multi-hop** | Single-hop similarity | GNN spreading activation (Eq 5) |
| **Recall** | Cosine similarity only | 5-head attention + BM25 hybrid (Eq 9, 19) |
| **Decay** | Manual TTL / none | ACT-R power-law decay (Eq 17) |
| **Cost** | $0.01-0.05 per memory op | $0.00 — numpy only |

### Key Capabilities

1. **Continuous Learning (No Fine-Tuning):** Memories have learned strengths and biases. Hebbian-like updates strengthen useful memories and decay unused ones — the system learns what matters.
2. **Adaptive Belief Updating:** Corrections trigger soft value drifts (gradient-descent inspired) that adjust memory representations in latent space, handling contradictions without deletion rules.
3. **Multi-Hop Synthesis via GNN:** A graph neural network connects memories by semantic/temporal proximity. Activation spreads through the graph, synthesizing disconnected facts (A knows B + B lives in C → A might visit C).
4. **5-Head Attention + BM25 Hybrid:** Semantic, answer-similarity, context, recency, and sparse lexical heads combine for recall that catches both meaning and exact keywords.
5. **Prospect Theory Gating:** Kahneman-inspired asymmetric weighting means gains (new useful info) are valued differently from losses (redundant info), improving what gets stored.
6. **Query-Aware Spreading:** GAT-style attention ensures only query-relevant memories participate in graph activation — well-connected but irrelevant memories can't steal signal.

---

## The 29 Equations

Every behavior in ITM is mathematically defined. No heuristics, no LLM classification, no rules.

### Phase 1: Core Memory (Perceptron + GNN)

| # | Name | Equation | Neural Network Origin |
|---|------|----------|----------------------|
| 1 | Strength Update | `s_i(t+1) = s_i·γ + α·sim(e_in, k_i)` | Perceptron weight update (1958) |
| 2 | Value Drift | `v_i(t+1) = v_i + b_i·(e_out_new - v_i)` | Gradient descent / bias adaptation |
| 3 | Recall | `R(q) = Σ(s_i·sim(q,k_i)·v_i) / Σ(s_i·sim)` | Weighted retrieval |
| 4 | Association Edge | `w_ij = (α_k·sim_k + α_v·sim_v)·exp(-\|Δt\|/τ)` | Hebbian learning (1949) |
| 5 | Spreading Activation | `a_i^(l+1) = a_i^(l) + η·Σ_j(w_ij·a_j^(l))` | GNN message passing (2017) |
| 6 | Memory Input Gate | `g = σ(β_g·((1-R_overlap) - θ_gate))` | LSTM input gate (1997) |
| 7 | Graph Contradiction | `v_j += b_j·(v_i - v_j), s_j *= δ` | Backpropagation through graph |

### Phase 1.5: Advanced Recall (RNN + Multi-Head Attention)

| # | Name | Equation | Neural Network Origin |
|---|------|----------|----------------------|
| 8 | Context Vector | `c_t = λ·c_{t-1} + (1-λ)·e_in_t` | RNN hidden state |
| 9 | Multi-Head Recall | 4 heads: semantic + answer + context + recency | Multi-head attention (2017) |
| 10 | Enhanced Gate | Primary + R_out/R_val/D_ctx penalties + hard ceiling | Deep gate networks |
| 11 | Adaptive Head Weights | Softmax over head confidences | Mixture of experts |

### Phase 2: Cognitive Science (Kahneman + ACT-R)

| # | Name | Equation | Origin |
|---|------|----------|--------|
| 12 | Prospect Strength | `Δs_gain·1.0` vs `Δs_loss·λ` (λ=2.25) | Kahneman prospect theory (1979) |
| 13 | Tension Detection | `T = 1 - max_sim` among recalled memories | WYSIATI-breaking conflict detection |
| 14 | Category Channels | fact / preference / instruction / event routing | Dual-process theory |
| 15 | Adaptive-k Recall | k scales with query difficulty | ACT-R retrieval threshold |
| 16 | Feedback Loop | Strengthen/weaken recalled based on response | Reinforcement learning |
| 17 | ACT-R Decay | `s_i *= (1 + t_idle)^(-d)` | ACT-R power-law decay (Anderson 1993) |

### Phase 3: MemWire-Inspired (ResNet + BM25 + JK-Net + GAT)

| # | Name | Equation | Neural Network Origin |
|---|------|----------|----------------------|
| 18 | Displacement Edges | `d_i = normalize(v_i - k_i)`, added to Eq 4 | ResNet residual connections (2015) |
| 19 | Sparse Lexical Recall | TF-IDF sparse vectors + BM25 scoring | BM25 (Robertson 1994) |
| 20 | Multi-Scale Activation | `a_final = Σ_l(w_l·a^(l))` with per-hop damping | JK-Net (Xu et al. 2018) |
| 21 | Query-Aware Spreading | `msg_ij = w_ij·a_j·sim(q, k_j)` | GAT attention (Veličković 2018) |

### Phase 4: Scaling & Robustness (Research.md §14-18)

| # | Name | Equation | Neural Network Origin |
|---|------|----------|----------------------|
| 22 | L2 Pattern Formation | `K_L2 = Σ(s_i·k_i)/Σ(s_i)`, `S_L2 = Σ(s_i)` | CNN weighted pooling (LeCun 1998) |
| 23 | Hierarchical Recall | `R(q) = w₁·R_L1 + w₂·R_L2 + w₃·R_L3` | ResNet skip connections (He 2015) |
| 24 | Multi-Channel Contradiction | `sim_mc = w_t·topic + w_d·disp + w_c·ctx` | Multi-head attention (Vaswani 2017) |
| 25 | Soft Create-vs-Strengthen | `p_str = σ(β·(sim - θ_local))` | Sigmoid activation (1986) |
| 26 | Adaptive Sharpness | `β = β_min + (β_max-β_min)·saturation` | Temperature scaling (softmax) |
| 27 | Adaptive Threshold | `θ_local = θ_base + β_θ·density(q)` | Adam optimizer / BatchNorm |
| 28 | Cold Start Boost | `α_eff = α·(1 + κ·exp(-n/τ))` | Inverted learning rate warmup |
| 29 | Shared Memory Merge | `R = (1-w_s)·R_personal + w_s·R_shared` | Federated averaging (McMahan 2017) |

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

Two lines to add memory to any OpenAI client:

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

# Use normally — memory is transparent
response = client.responses.create(
    model="gpt-4.1-nano",
    input="Hi, I'm Harshal. I study at VIT."
)
print(response.output_text)

# Memory update runs in background thread — zero latency added
mem.flush()
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
User Input → LLM (frozen) → Response
     ↓                          ↓
  embed(input)            embed(output)
     ↓                          ↓
  ┌──────────────────────────────────────────┐
  │           MEMORY LAYER (29 equations)    │
  │                                          │
  │  Phase 1: Core                           │
  │    Eq 1-3: Strength / Value / Recall     │
  │    Eq 4-5: Graph Edges + GNN Spreading   │
  │    Eq 6-7: LSTM Gate + Graph Backprop    │
  │                                          │
  │  Phase 1.5: Advanced Recall              │
  │    Eq 8: RNN Context Vector              │
  │    Eq 9: Multi-Head Attention (5 heads)  │
  │    Eq 10-11: Enhanced Gate + Adaptive    │
  │                                          │
  │  Phase 2: Cognitive Science              │
  │    Eq 12: Prospect Theory Gating         │
  │    Eq 13-14: Tension + Channels          │
  │    Eq 15-17: Adaptive-k + ACT-R Decay   │
  │                                          │
  │  Phase 3: MemWire-Inspired               │
  │    Eq 18: ResNet Displacement Edges      │
  │    Eq 19: BM25 Sparse Lexical Recall     │
  │    Eq 20: JK-Net Multi-Scale Activation  │
  │    Eq 21: GAT Query-Aware Spreading      │
  └──────────────────────────────────────────┘
```

## Project Structure

```
itm/
  config.py          — All hyperparameters (29 equations worth)
  core.py            — MemoryEntry + MemoryLayer + SparseIndex
  graph.py           — MemoryGraph (GNN edges + spreading activation)
  hierarchy.py       — L1/L2/L3 consolidation + hierarchical recall
  embeddings.py      — BGE-M3 local embeddings (on-device, free)
  embeddings_api.py  — Optional alternate backends (Nomic, OpenAI, Cohere)
  storage.py         — Save/load to disk (.npz + .json)
  patch.py           — Transparent OpenAI client adapter
  formatting.py      — Memory display + prompt formatting
  stats.py           — CLI statistics and graph search

tests/               — 179 tests across 7 test files
gpt.py               — Interactive chat demo with full memory
test_gpt.py          — Real-BGE-M3 integration test (not run in CI)
```

---

## Performance

- **179 tests passing** across unit and integration suites
- **10/11 recall accuracy** on personal fact retrieval (integration test)
- **Zero LLM calls** for all memory operations
- **Background processing** — memory updates run in thread pool, zero added latency
- **On-device embeddings** — BGE-M3 via sentence-transformers, no API costs

---

## vs. The Market

| System | Approach | LLM Calls for Memory | Learning | Multi-Hop |
|--------|----------|---------------------|----------|-----------|
| **ITM** | 29 equations, pure math | 0 | Continuous (Hebbian) | GNN + GAT |
| mem0 | LLM extracts + classifies | 2-3 per interaction | None (static store) | None |
| RAG | Chunk + embed + retrieve | 0 (but no learning) | None | None |
| Advanced RAG | Rerank + hybrid search | 1+ for reranking | None | Limited |
| MemGPT | LLM manages own memory | 3-5 per interaction | LLM-simulated | LLM-simulated |
| Zep | Summary + entity extraction | 1-2 per interaction | None | Graph (LLM-built) |
| LangChain Memory | Buffer/summary/entity | 1-2 for summary | None | None |

**ITM's differentiator**: It's the only system where memory operations are *learned behaviors* governed by mathematical equations, not LLM-orchestrated storage operations.

---

## Research

The theoretical foundations span 70 years of neural network history — from
Rosenblatt's Perceptron (1958) to Graph Attention Networks (2018) — with each
equation mapped to its neural-network origin. See the equation tables above for
the full mapping.

---

## Running Tests

```bash
# All unit tests (179 tests, offline, deterministic FakeEmbedder)
pip install -e .[dev]
python -m pytest tests/ -v

# Integration test with real BGE-M3 embeddings (downloads model, needs network)
python test_gpt.py
```

---

## License

[MIT](LICENSE) — Copyright (c) 2026 Harshal More.

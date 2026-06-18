"""LOCOMO Benchmark Evaluation for Inference-Time Learning Memory Layer.

Evaluates the memory system against the LOCOMO benchmark (ACL 2024)
for long-term conversational memory in LLMs.

Pipeline per conversation:
  1. INGEST — Feed conversation turns into a fresh MemoryLayer
  2. QUERY  — For each QA question, recall_graph() → memory context
  3. ANSWER — Memory context + question → LLM → short answer
  4. SCORE  — Token F1 (original LOCOMO) + LLM-as-Judge (modern)
  5. REPORT — Per-category and overall accuracy

Usage:
  python benchmarks/evaluate_memory.py \\
    --data-file benchmarks/locomo/data/locomo10.json \\
    --model gpt-4.1-nano

  # Single conversation for debugging
  python benchmarks/evaluate_memory.py \\
    --data-file benchmarks/locomo/data/locomo10.json \\
    --model gpt-4.1-nano --sample-id conv-26 --verbose
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from collections import Counter
import string
import regex

from itm.config import MemoryConfig
from itm.core import MemoryLayer
from itm.embeddings import EmbeddingService
from itm.formatting import MemoryFormatter

# ── LOCOMO scoring functions (inlined to avoid bert_score dependency) ──
# These are the exact implementations from LOCOMO's evaluation.py

from nltk.stem import PorterStemmer

_ps = PorterStemmer()


def _normalize_answer(s: str) -> str:
    """Normalize answer string (from LOCOMO evaluation.py)."""
    s = s.replace(",", "")
    s = s.lower()
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Remove articles
    s = regex.sub(r"\b(a|an|the|and)\b", " ", s)
    # Fix whitespace
    s = " ".join(s.split())
    return s


def locomo_f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 with Porter stemming (from LOCOMO evaluation.py)."""
    pred_tokens = [_ps.stem(w) for w in _normalize_answer(prediction).split()]
    truth_tokens = [_ps.stem(w) for w in _normalize_answer(ground_truth).split()]
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def locomo_f1_multi(prediction: str, ground_truth: str) -> float:
    """Multi-answer F1: split by comma, compute partial F1 (from LOCOMO)."""
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return float(
        np.mean(
            [
                max([locomo_f1_score(pred, gt) for pred in predictions])
                for gt in ground_truths
            ]
        )
    )


load_dotenv()

# ── Category names for reporting ──

CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

# ── Prompts ──

SYSTEM_PROMPT = (
    "You are answering questions about a conversation between two people. "
    "Use the provided memory context to answer. Give short, precise answers "
    "using exact words from the context whenever possible. "
    "If the information is not available, say 'Not mentioned in the conversation'."
)

QA_PROMPT = (
    "Based on the memory context above, write a short answer (a few words) "
    "for the following question. Answer with exact words from the context.\n\n"
    "Question: {question}\nShort answer:"
)

QA_PROMPT_TEMPORAL = (
    "Based on the memory context above, write a short answer with an approximate "
    "date for the following question. Use the dates mentioned in the context.\n\n"
    "Question: {question}\nShort answer:"
)

QA_PROMPT_ADVERSARIAL = (
    "Based on the memory context above, answer the following question. "
    "If the information was NOT mentioned in the conversation, respond with "
    "'Not mentioned in the conversation'.\n\n"
    "Question: {question}\nShort answer:"
)

JUDGE_PROMPT = (
    "You are a judge evaluating whether a predicted answer is correct.\n\n"
    "Question: {question}\n"
    "Ground truth answer: {answer}\n"
    "Predicted answer: {prediction}\n\n"
    "Is the predicted answer correct? It doesn't need to be an exact match, "
    "but it should convey the same information as the ground truth.\n"
    "Respond with exactly one word: CORRECT or WRONG"
)


def _base_kwargs() -> dict:
    """Full 29-equation substrate, identical to the diag scripts and gpt.py.

    The three ablation configs differ ONLY in transcript_ingest (Fix A) and
    spreading_debias_enabled (Fix B); everything below is held constant so the
    answer-accuracy ablation slope is comparable to the offline recall@5 slope.
    """
    return dict(
        prospect_enabled=True,
        tension_enabled=True,
        channels_enabled=True,
        adaptive_k_enabled=True,
        feedback_enabled=True,
        actr_enabled=True,
        displacement_edges_enabled=True,
        sparse_recall_enabled=True,
        multiscale_enabled=True,
        query_aware_spread_enabled=True,
        hierarchy_enabled=True,
        multichannel_enabled=True,
        soft_decisions_enabled=True,
        cold_start_enabled=True,
    )


# ═══════════════════════════════════════════════════════════
# INGESTION — Feed conversation turns into memory
# ═══════════════════════════════════════════════════════════


def get_session_numbers(conversation: dict) -> list[int]:
    """Extract sorted session numbers from conversation dict."""
    nums = []
    for key in conversation:
        if key.startswith("session_") and "date_time" not in key:
            try:
                nums.append(int(key.split("_")[-1]))
            except ValueError:
                continue
    return sorted(nums)


def _collect_turns(conversation: dict) -> list[dict]:
    """Collect all turns in chronological order with timestamps."""
    session_nums = get_session_numbers(conversation)
    all_turns = []
    for session_num in session_nums:
        date_time = conversation.get(f"session_{session_num}_date_time", "")
        session_turns = conversation.get(f"session_{session_num}", [])
        for turn in session_turns:
            text = f"[{date_time}] {turn['speaker']}: {turn['text']}"
            if "blip_caption" in turn and turn["blip_caption"]:
                text += f" [shared image: {turn['blip_caption']}]"
            all_turns.append(
                {
                    "text": text,
                    "date_time": date_time,
                    "speaker": turn["speaker"],
                    "raw_text": turn["text"],
                    "dia_id": turn.get("dia_id", ""),
                }
            )
    return all_turns


def ingest_conversation(
    layer: MemoryLayer,
    embedder: EmbeddingService,
    conversation: dict,
    verbose: bool = False,
) -> dict:
    """Feed all conversation turns into memory using sliding window.

    Uses the full update() pipeline (Equations 1-11) including gate filtering.
    Returns stats dict with counts of actions taken.
    """
    action_counts = defaultdict(int)
    all_turns = _collect_turns(conversation)
    total_turns = len(all_turns)

    if total_turns == 0:
        return {"total_turns": 0, "memories_stored": 0, "actions": {}}

    for i in range(total_turns):
        input_text = all_turns[i]["text"]
        if i + 1 < total_turns:
            output_text = all_turns[i + 1]["text"]
        else:
            output_text = f"[End of conversation on {all_turns[i]['date_time']}]"

        input_emb = embedder.embed(input_text)
        output_emb = embedder.embed(output_text)
        result = layer.update(input_emb, output_emb, input_text, output_text)
        action_counts[result["action"]] += 1

        if verbose and i < 5:
            print(f"  Turn {i+1}: {result['action']} | {input_text[:80]}...")

    if verbose:
        print(f"  ... ({total_turns} turns total)")
        print(f"  Actions: {dict(action_counts)}")
        print(f"  Memories stored: {len(layer.memories)}")

    return {
        "total_turns": total_turns,
        "memories_stored": len(layer.memories),
        "actions": dict(action_counts),
    }


def ingest_conversation_direct(
    layer: MemoryLayer,
    embedder: EmbeddingService,
    conversation: dict,
    verbose: bool = False,
) -> dict:
    """Directly store EVERY turn as a memory — bypass gate/update entirely.

    This isolates recall quality (Equations 3, 8-9-11, 4-5) from
    storage filtering (Equations 6, 10, theta_create).

    Each turn gets its own MemoryEntry with:
      key   = embed(turn text)
      value = embed(next turn text)  [sliding window]
    """
    from itm.core import MemoryEntry

    all_turns = _collect_turns(conversation)
    total_turns = len(all_turns)

    if total_turns == 0:
        return {"total_turns": 0, "memories_stored": 0, "actions": {"direct": 0}}

    # Batch embed all turns for efficiency
    all_texts = [t["text"] for t in all_turns]
    all_embeddings = embedder.embed_batch(all_texts)

    # Also embed the "end" marker for the last turn's value
    end_text = f"[End of conversation on {all_turns[-1]['date_time']}]"
    end_emb = embedder.embed(end_text)

    for i in range(total_turns):
        input_emb = all_embeddings[i]
        input_text = all_turns[i]["text"]

        if i + 1 < total_turns:
            output_emb = all_embeddings[i + 1]
            output_text = all_turns[i + 1]["text"]
        else:
            output_emb = end_emb
            output_text = end_text

        # Directly create MemoryEntry — NO gate, NO threshold, NO decay
        entry = MemoryEntry(
            key=input_emb.copy(),
            value=output_emb.copy(),
            strength=layer.config.initial_strength,
            bias=layer.config.initial_bias,
            timestamp=i + 1,
            access_count=0,
            input_text=input_text,
            output_text=output_text,
        )
        layer.memories.append(entry)
        layer.timestep = i + 1

        # Equation 19: keep the sparse lexical index aligned with memories.
        # recall_graph computes h5 over sparse_index; if it stays empty while
        # sparse_recall_enabled is True the head shapes mismatch (h5 is (0,)).
        if layer.config.sparse_recall_enabled:
            layer.sparse_index.add(input_text)

        # Build graph edges (Equation 4)
        new_idx = len(layer.memories) - 1
        layer.graph.add_memory(new_idx, layer.memories)

    if verbose:
        print(f"  Direct ingest: {total_turns} turns → {len(layer.memories)} memories")
        print(f"  Graph edges built for all memories")

    return {
        "total_turns": total_turns,
        "memories_stored": len(layer.memories),
        "actions": {"direct": total_turns},
    }


# ═══════════════════════════════════════════════════════════
# ANSWERING — Recall memories + LLM answer
# ═══════════════════════════════════════════════════════════


def answer_question(
    layer: MemoryLayer,
    embedder: EmbeddingService,
    formatter: MemoryFormatter,
    client: OpenAI,
    question: str,
    category: int,
    model: str,
    verbose: bool = False,
) -> tuple[str, int, list]:
    """Recall memories for a question and get LLM answer.

    Returns (answer_text, num_memories_recalled, recalled_pairs).
    """
    # Recall relevant memories via graph spreading activation.
    # query_text is required so the sparse lexical head (Eq 19) fires — this
    # matches the offline recall@5 measurement exactly (apples-to-apples).
    query_emb = embedder.embed(question)
    recalled = layer.recall_graph(query_emb, query_text=question)

    # Limit to top_k (recall_graph returns all above activation_threshold)
    top_k = layer.config.top_k
    if len(recalled) > top_k:
        recalled = recalled[:top_k]

    memory_context = formatter.format_for_prompt(recalled)

    # Build category-specific prompt
    if category == 2:
        qa_prompt = QA_PROMPT_TEMPORAL.format(question=question)
    elif category == 5:
        qa_prompt = QA_PROMPT_ADVERSARIAL.format(question=question)
    else:
        qa_prompt = QA_PROMPT.format(question=question)

    # Combine memory context + question
    full_input = f"{memory_context}\n\n{qa_prompt}" if memory_context else qa_prompt

    if verbose:
        print(f"  Recalled {len(recalled)} memories")
        for mem, score in recalled[:3]:
            print(f"    [{score:.3f}] {mem.input_text[:80]}")

    # Call LLM
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=full_input,
    )

    answer = (getattr(response, "output_text", "") or "").strip()
    return answer, len(recalled), recalled


# ═══════════════════════════════════════════════════════════
# SCORING — Token F1 + LLM-as-Judge
# ═══════════════════════════════════════════════════════════


def score_f1(prediction: str, answer: str, category: int) -> float:
    """Score a single prediction using LOCOMO's exact metrics."""
    answer = str(answer)

    if category in [2, 3, 4]:
        # Standard token F1
        if category == 3:
            answer = answer.split(";")[0].strip()
        return locomo_f1_score(prediction, answer)

    elif category == 1:
        # Multi-hop: split by comma, partial F1
        return locomo_f1_multi(prediction, answer)

    elif category == 5:
        # Adversarial: check for refusal
        pred_lower = prediction.lower()
        if "no information available" in pred_lower or "not mentioned" in pred_lower:
            return 1.0
        return 0.0

    return 0.0


def score_judge(
    client: OpenAI,
    question: str,
    answer: str,
    prediction: str,
    model: str = "gpt-4.1-nano",
) -> float:
    """LLM-as-Judge scoring: CORRECT (1.0) or WRONG (0.0)."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=str(answer),
        prediction=prediction,
    )

    try:
        response = client.responses.create(
            model=model,
            instructions="You are a strict but fair judge. Respond with exactly one word.",
            input=prompt,
        )
        result = (getattr(response, "output_text", "") or "").strip().upper()
        return 1.0 if "CORRECT" in result else 0.0
    except Exception as e:
        print(f"  Judge error: {e}")
        return 0.0


# ═══════════════════════════════════════════════════════════
# REPORTING — Format results for terminal + JSON
# ═══════════════════════════════════════════════════════════


def print_report(results: dict):
    """Print formatted benchmark results to terminal."""
    by_cat = results["by_category"]

    print("\n" + "=" * 70)
    print("LOCOMO BENCHMARK RESULTS — Inference-Time Learning Memory Layer")
    print("=" * 70)

    # Header
    print(f"\n{'Category':<20} {'Count':>6} {'F1':>8} {'Judge':>8}")
    print("-" * 44)

    # Per-category rows
    cat_order = [4, 1, 2, 3, 5]  # LOCOMO's display order
    for cat in cat_order:
        name = CATEGORY_NAMES.get(cat, str(cat))
        info = by_cat.get(str(cat), {})
        count = info.get("count", 0)
        f1 = info.get("f1", 0.0)
        judge = info.get("judge", 0.0)
        print(f"  {name:<18} {count:>6} {f1:>7.3f} {judge:>7.3f}")

    print("-" * 44)

    # Overall
    overall_f1 = results.get("overall_f1", 0.0)
    overall_judge = results.get("overall_judge", 0.0)
    total_count = sum(
        by_cat[str(c)].get("count", 0) for c in cat_order if str(c) in by_cat
    )
    print(
        f"  {'OVERALL':<18} {total_count:>6} {overall_f1:>7.3f} {overall_judge:>7.3f}"
    )

    # Memory stats
    mem_stats = results.get("memory_stats", {})
    if mem_stats:
        print(f"\n{'Memory Stats':}")
        print(
            f"  Avg memories/conversation:      {mem_stats.get('avg_memories_per_conv', 0):.1f}"
        )
        print(
            f"  Avg memories recalled/question:  {mem_stats.get('avg_recalled_per_question', 0):.1f}"
        )
        print(
            f"  Avg turns filtered (%):          {mem_stats.get('avg_filtered_pct', 0):.1f}%"
        )

    print("=" * 70)

    # Comparison table
    print("\nComparison with published baselines (LLM-as-Judge):")
    print(f"  {'System':<25} {'Overall':>8}")
    print(f"  {'-'*35}")
    baselines = [
        ("Mem0 (graph)", "~66.9%"),
        ("Letta (filesystem)", "~74.0%"),
        ("Zep", "~75.1%"),
        ("Our Memory Layer", f"{overall_judge*100:.1f}%"),
    ]
    for name, score in baselines:
        print(f"  {name:<25} {score:>8}")
    print()


# ═══════════════════════════════════════════════════════════
# MAIN — Orchestrate the full benchmark
# ═══════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(description="LOCOMO benchmark for memory system")
    parser.add_argument("--data-file", required=True, help="Path to locomo10.json")
    parser.add_argument("--model", default="gpt-4.1-nano", help="LLM for answering")
    parser.add_argument("--judge-model", default="gpt-4.1-nano", help="LLM for judging")
    parser.add_argument(
        "--out-dir", default="benchmarks/results", help="Output directory"
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Run only these conversations (comma-separated for a multi-conv slice)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override recall top_k")
    parser.add_argument(
        "--n-hops", type=int, default=None, help="Override graph n_hops"
    )
    parser.add_argument(
        "--theta-create", type=float, default=None, help="Override theta_create"
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="Skip LLM-as-Judge scoring"
    )
    parser.add_argument(
        "--recall-only",
        action="store_true",
        help="Bypass gate/filtering — store every turn directly, test recall quality only",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Cap questions per conversation (sanity/cost control)",
    )
    parser.add_argument(
        "--answerable-only",
        action="store_true",
        help="Skip category 5 (adversarial) questions entirely",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show per-question details"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load LOCOMO data
    print(f"Loading LOCOMO data from {args.data_file}")
    with open(args.data_file) as f:
        samples = json.load(f)
    print(f"  {len(samples)} conversations loaded")

    # Filter to a held-out slice if requested (one id or comma-separated list)
    if args.sample_id:
        wanted = [s.strip() for s in args.sample_id.split(",") if s.strip()]
        samples = [s for s in samples if s["sample_id"] in wanted]
        found = {s["sample_id"] for s in samples}
        missing = [w for w in wanted if w not in found]
        if missing:
            print(f"ERROR: sample_id(s) not found: {missing}")
            sys.exit(1)
        print(f"  Filtered to: {sorted(found)}")

    # Build memory config on the full 29-equation substrate (matches gpt.py and
    # the offline recall@5 diag exactly, so answer accuracy is apples-to-apples
    # with the 3%->38%->64% recall numbers). Fix A (transcript_ingest) and Fix B
    # (spreading_debias_enabled) are toggled via env so the diag scripts and this
    # harness share one knob:
    #   baseline: ITM_TRANSCRIPT_INGEST=0 ITM_DEBIAS=0
    #   +A:       ITM_TRANSCRIPT_INGEST=1 ITM_DEBIAS=0
    #   +A+B:     ITM_TRANSCRIPT_INGEST=1 ITM_DEBIAS=1
    config_kwargs = _base_kwargs()
    config_kwargs["transcript_ingest"] = (
        os.environ.get("ITM_TRANSCRIPT_INGEST", "0") == "1"
    )
    config_kwargs["spreading_debias_enabled"] = (
        os.environ.get("ITM_DEBIAS", "0") == "1"
    )
    if args.top_k is not None:
        config_kwargs["top_k"] = args.top_k
    if args.n_hops is not None:
        config_kwargs["n_hops"] = args.n_hops
    if args.theta_create is not None:
        config_kwargs["theta_create"] = args.theta_create
    config = MemoryConfig(**config_kwargs)
    print(
        f"Config: transcript_ingest={config.transcript_ingest} "
        f"spreading_debias_enabled={config.spreading_debias_enabled}"
    )

    # Initialize shared embedder (BGE-M3, loaded once)
    print("Loading BGE-M3 embedding model...")
    embedder = EmbeddingService(config)
    formatter = MemoryFormatter()

    # Initialize OpenAI client. Bump max_retries (default 2) so transient rate
    # limits / 5xx are absorbed with exponential backoff rather than dropping a
    # question on a long full-dataset run.
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_retries=8,
    )

    # ── Run evaluation ──
    all_results = []  # per-question results
    conv_stats = []  # per-conversation memory stats

    # Incremental checkpoint: write per-question results to the same path the
    # final report uses, after every conversation, so a crash/timeout on a long
    # run never loses completed work.
    tag = f"A{int(config.transcript_ingest)}B{int(config.spreading_debias_enabled)}"
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, f"locomo_memory_results_{tag}.json")

    for sample in samples:
        sample_id = sample["sample_id"]
        conversation = sample["conversation"]
        qa_list = sample["qa"]

        print(f"\n{'='*60}")
        print(f"Conversation: {sample_id} ({len(qa_list)} questions)")
        print(f"{'='*60}")

        # Fresh memory layer per conversation (no cross-contamination)
        layer = MemoryLayer(config, embedder)

        # Clear embedder cache between conversations to avoid memory bloat
        embedder._cache.clear()

        # 1. INGEST conversation
        if args.recall_only:
            print(
                "Ingesting conversation turns (RECALL-ONLY: bypass gate, store all)..."
            )
            ingest_stats = ingest_conversation_direct(
                layer, embedder, conversation, verbose=args.verbose
            )
        else:
            print("Ingesting conversation turns...")
            ingest_stats = ingest_conversation(
                layer, embedder, conversation, verbose=args.verbose
            )
        conv_stats.append(ingest_stats)

        # Map memory input_text -> dia_id so we can report whether gold evidence
        # was actually present in the recalled top-k context (no extra API cost).
        text_to_dia = {t["text"]: t["dia_id"] for t in _collect_turns(conversation)}

        # Optional filtering for sanity / cost control
        if args.answerable_only:
            qa_list = [q for q in qa_list if q["category"] != 5]
        if args.max_questions is not None:
            qa_list = qa_list[: args.max_questions]

        # 2-3. QUERY + ANSWER each question
        print(f"Answering {len(qa_list)} questions...")
        for i, qa in enumerate(tqdm(qa_list, desc=f"  {sample_id}")):
            question = qa["question"]
            category = qa["category"]

            # Category 5 adversarial questions use 'adversarial_answer'
            # (the correct response is "Not mentioned"; adversarial_answer
            # is the trap answer that was never actually discussed)
            if "answer" in qa:
                answer = str(qa["answer"])
            elif "adversarial_answer" in qa:
                # For adversarial Qs, the ground truth is that it's NOT in conversation
                answer = "Not mentioned in the conversation"
            else:
                print(f"  WARNING: QA missing answer field, skipping: {question[:50]}")
                continue

            # Get prediction
            prediction, num_recalled, recalled = answer_question(
                layer,
                embedder,
                formatter,
                client,
                question,
                category,
                args.model,
                verbose=args.verbose,
            )

            # Was at least one gold evidence turn present in the recalled top-k?
            evidence = qa.get("evidence", []) or []
            recalled_dia = {
                text_to_dia.get(m.input_text) for m, _ in recalled
            }
            gold_present = bool(evidence) and any(
                d in recalled_dia for d in evidence
            )

            # 4. SCORE — F1
            f1_val = score_f1(prediction, answer, category)

            # 4. SCORE — Judge
            judge_val = 0.0
            if not args.no_judge:
                judge_val = score_judge(
                    client, question, answer, prediction, args.judge_model
                )

            result = {
                "sample_id": sample_id,
                "question": question,
                "answer": answer,
                "category": category,
                "prediction": prediction,
                "f1": round(f1_val, 4),
                "judge": judge_val,
                "num_recalled": num_recalled,
                "gold_present": gold_present,
                "has_evidence": bool(evidence),
            }
            all_results.append(result)

            if args.verbose:
                status = "OK" if f1_val > 0.5 else "MISS"
                print(
                    f"  [{status}] Cat{category} F1={f1_val:.3f} J={judge_val:.0f} "
                    f"| Q: {question[:50]}..."
                )
                print(f"         A: {answer[:50]} | P: {prediction[:50]}")

        # Checkpoint after each conversation so progress survives a crash/timeout.
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(
            f"  [checkpoint] {len(all_results)} questions scored so far "
            f"-> {results_path}"
        )

    # ── Aggregate results ──
    print("\nAggregating results...")

    by_category = {}
    for cat in [1, 2, 3, 4, 5]:
        cat_results = [r for r in all_results if r["category"] == cat]
        if cat_results:
            by_category[str(cat)] = {
                "count": len(cat_results),
                "f1": round(np.mean([r["f1"] for r in cat_results]), 4),
                "judge": round(np.mean([r["judge"] for r in cat_results]), 4),
                "avg_recalled": round(
                    np.mean([r["num_recalled"] for r in cat_results]), 2
                ),
            }

    overall_f1 = round(np.mean([r["f1"] for r in all_results]), 4)
    overall_judge = round(np.mean([r["judge"] for r in all_results]), 4)

    # Answerable-only (cats 1-4): the prior 24.4% "overall" was a refusal artifact
    # carried by the cat5 adversarial set, so report this separately.
    ans = [r for r in all_results if r["category"] != 5]
    answerable = {
        "count": len(ans),
        "f1": round(np.mean([r["f1"] for r in ans]), 4) if ans else 0.0,
        "judge": round(np.mean([r["judge"] for r in ans]), 4) if ans else 0.0,
        "gold_present_pct": (
            round(100.0 * np.mean([r["gold_present"] for r in ans]), 1) if ans else 0.0
        ),
    }
    # Accuracy split by whether gold evidence was in the recalled context.
    ans_gold = [r for r in ans if r["gold_present"]]
    ans_nogold = [r for r in ans if not r["gold_present"]]
    answerable["judge_when_gold_present"] = (
        round(np.mean([r["judge"] for r in ans_gold]), 4) if ans_gold else None
    )
    answerable["judge_when_gold_absent"] = (
        round(np.mean([r["judge"] for r in ans_nogold]), 4) if ans_nogold else None
    )
    print(
        f"\nANSWERABLE-ONLY (cats 1-4): n={answerable['count']} "
        f"judge={answerable['judge']:.4f} f1={answerable['f1']:.4f} "
        f"gold_present={answerable['gold_present_pct']}%  "
        f"judge|gold_present={answerable['judge_when_gold_present']} "
        f"judge|gold_absent={answerable['judge_when_gold_absent']}"
    )

    # Memory stats
    avg_memories = (
        np.mean([s["memories_stored"] for s in conv_stats]) if conv_stats else 0
    )
    avg_recalled = (
        np.mean([r["num_recalled"] for r in all_results]) if all_results else 0
    )
    avg_filtered = 0.0
    if conv_stats:
        filtered_counts = [s["actions"].get("filtered", 0) for s in conv_stats]
        total_counts = [s["total_turns"] for s in conv_stats]
        if sum(total_counts) > 0:
            avg_filtered = 100.0 * sum(filtered_counts) / sum(total_counts)

    report = {
        "mode": "recall_only" if args.recall_only else "full_pipeline",
        "model": args.model,
        "judge_model": args.judge_model if not args.no_judge else None,
        "memory_config": {
            "top_k": config.top_k,
            "n_hops": config.n_hops,
            "theta_create": config.theta_create,
            "beta_gate": config.beta_gate,
            "theta_gate": config.theta_gate,
            "embedding_model": config.embedding_model,
            "transcript_ingest": config.transcript_ingest,
            "spreading_debias_enabled": config.spreading_debias_enabled,
        },
        "overall_f1": overall_f1,
        "overall_judge": overall_judge,
        "answerable": answerable,
        "by_category": by_category,
        "memory_stats": {
            "avg_memories_per_conv": round(avg_memories, 1),
            "avg_recalled_per_question": round(avg_recalled, 1),
            "avg_filtered_pct": round(avg_filtered, 1),
        },
    }

    # ── Save results ──
    # Tag filenames with the ablation config so baseline/+A/+A+B runs do not
    # overwrite each other.
    tag = (
        f"A{int(config.transcript_ingest)}B{int(config.spreading_debias_enabled)}"
    )
    os.makedirs(args.out_dir, exist_ok=True)

    results_path = os.path.join(args.out_dir, f"locomo_memory_results_{tag}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Per-question results saved to {results_path}")

    stats_path = os.path.join(args.out_dir, f"locomo_memory_stats_{tag}.json")
    with open(stats_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Aggregate stats saved to {stats_path}")

    # ── Print report ──
    print_report(report)


if __name__ == "__main__":
    main()

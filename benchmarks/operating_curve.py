"""Abstention operating curve for ITM on LOCOMO.

A single accuracy number on a benchmark that mixes answerable and unanswerable
questions is uninterpretable: a system that refuses everything scores ~100% on
the adversarial split for free. This harness sweeps the retrieval budget -- the
amount of evidence that reaches the reader -- and reports where the system sits
on the answerable-vs-adversarial frontier at each operating point.

k=0 is included deliberately: it is the empty-context anchor, i.e. the
always-refuse policy, which is the floor any reported adversarial accuracy must
be judged against.

Retrieval is computed ONCE per question at the largest k in the sweep and then
truncated, so the sweep costs one ingest pass plus two model calls per
(question, k). Embeddings are cached on disk across runs.

Usage:
    python benchmarks/operating_curve.py \
        --data-file benchmarks/locomo/data/locomo10.json \
        --answerable 250 --adversarial 150

    # every question, no subsampling
    python benchmarks/operating_curve.py --data-file ... --full
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.evaluate_memory import (  # noqa: E402
    CATEGORY_NAMES,
    QA_PROMPT,
    QA_PROMPT_ADVERSARIAL,
    QA_PROMPT_TEMPORAL,
    SYSTEM_PROMPT,
    _base_kwargs,
    _collect_turns,
    ingest_conversation,
    score_judge,
)
from itm.config import MemoryConfig  # noqa: E402
from itm.core import MemoryLayer  # noqa: E402
from itm.embeddings import EmbeddingService  # noqa: E402
from itm.formatting import MemoryFormatter  # noqa: E402

DEFAULT_SWEEP = [0, 1, 2, 3, 5, 8, 12, 20, 30]
REFUSAL_MARKERS = ("not mentioned", "no information available")


# ═══════════════════════════════════════════════════════════
# EMBEDDING CACHE — the sweep is re-run often; embedding is the slow part
# ═══════════════════════════════════════════════════════════


class CachedEmbedder:
    """Wraps EmbeddingService with a disk-backed cache keyed by text hash."""

    def __init__(self, config: MemoryConfig, cache_path: Path):
        self._inner = EmbeddingService(config)
        self._path = cache_path
        self._cache: dict[str, np.ndarray] = {}
        if cache_path.exists():
            try:
                self._cache = pickle.loads(cache_path.read_bytes())
            except Exception:
                self._cache = {}
        self._dirty = 0

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> np.ndarray:
        k = self._key(text)
        if k not in self._cache:
            self._cache[k] = self._inner.embed(text)
            self._dirty += 1
        return self._cache[k]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        missing = [t for t in texts if self._key(t) not in self._cache]
        if missing:
            for text, vec in zip(missing, self._inner.embed_batch(missing)):
                self._cache[self._key(text)] = vec
                self._dirty += 1
        return [self._cache[self._key(t)] for t in texts]

    def flush(self) -> None:
        if self._dirty:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_bytes(pickle.dumps(self._cache))
            self._dirty = 0


# ═══════════════════════════════════════════════════════════
# SAMPLING
# ═══════════════════════════════════════════════════════════


def select_questions(samples: list[dict], n_ans: int, n_adv: int, seed: int) -> dict:
    """Stratified subsample: n_ans across categories 1-4, n_adv from category 5."""
    rng = random.Random(seed)
    pool_ans, pool_adv = [], []
    for si, sample in enumerate(samples):
        for qi, qa in enumerate(sample.get("qa", [])):
            cat = qa.get("category")
            ref = (si, qi)
            if cat in (1, 2, 3, 4):
                pool_ans.append(ref)
            elif cat == 5:
                pool_adv.append(ref)

    if n_ans > 0:
        by_cat = defaultdict(list)
        for si, qi in pool_ans:
            by_cat[samples[si]["qa"][qi]["category"]].append((si, qi))
        chosen: list[tuple[int, int]] = []
        for cat in sorted(by_cat):
            share = max(1, round(n_ans * len(by_cat[cat]) / len(pool_ans)))
            chosen.extend(rng.sample(by_cat[cat], min(share, len(by_cat[cat]))))
        pool_ans = chosen[:n_ans] if len(chosen) > n_ans else chosen
    if n_adv > 0:
        pool_adv = rng.sample(pool_adv, min(n_adv, len(pool_adv)))

    selected = defaultdict(list)
    for si, qi in pool_ans + pool_adv:
        selected[si].append(qi)
    return selected


# ═══════════════════════════════════════════════════════════
# RETRIEVAL — one pass, ranked to max(sweep)
# ═══════════════════════════════════════════════════════════


def retrieve_all(
    layer: MemoryLayer,
    embedder: CachedEmbedder,
    sample: dict,
    q_indices: list[int],
    max_k: int,
    text_to_dia: dict,
) -> list[dict]:
    """Recall once per question, keeping the top max_k ranked memories."""
    out = []
    qas = sample["qa"]
    questions = [str(qas[qi]["question"]) for qi in q_indices]
    q_embs = embedder.embed_batch(questions)

    for qi, question, q_emb in zip(q_indices, questions, q_embs):
        qa = qas[qi]
        recalled = layer.recall_graph(np.asarray(q_emb), query_text=question)[:max_k]
        dias = [text_to_dia.get(m.input_text) for m, _ in recalled]
        out.append(
            {
                "sample_id": sample.get("sample_id"),
                "question": question,
                "answer": qa.get("answer", qa.get("adversarial_answer", "")),
                "category": qa.get("category"),
                "evidence": qa.get("evidence") or [],
                "recalled": recalled,
                "recalled_dias": dias,
            }
        )
    return out


# ═══════════════════════════════════════════════════════════
# ANSWER + JUDGE at one operating point
# ═══════════════════════════════════════════════════════════


def answer_one(
    client: OpenAI,
    formatter: MemoryFormatter,
    item: dict,
    k: int,
    model: str,
    judge_model: str,
) -> dict:
    cat = item["category"]
    context = formatter.format_for_prompt(item["recalled"][:k]) if k > 0 else ""

    if cat == 2:
        qa_prompt = QA_PROMPT_TEMPORAL.format(question=item["question"])
    elif cat == 5:
        qa_prompt = QA_PROMPT_ADVERSARIAL.format(question=item["question"])
    else:
        qa_prompt = QA_PROMPT.format(question=item["question"])
    full_input = f"{context}\n\n{qa_prompt}" if context else qa_prompt

    try:
        resp = client.responses.create(
            model=model, instructions=SYSTEM_PROMPT, input=full_input
        )
        prediction = (getattr(resp, "output_text", "") or "").strip()
    except Exception as exc:  # keep the sweep alive on transient failures
        return {"error": str(exc), "category": cat}

    low = prediction.lower()
    refused = any(m in low for m in REFUSAL_MARKERS)

    if cat == 5:
        # Refusal IS the correct answer; no judge call needed.
        judge = 1.0 if refused else 0.0
    else:
        judge = score_judge(
            client, item["question"], item["answer"], prediction, judge_model
        )

    gold = bool(item["evidence"]) and any(
        d in set(item["recalled_dias"][:k]) for d in item["evidence"]
    )
    return {
        "category": cat,
        "judge": judge,
        "refused": refused,
        "gold_present": gold,
        "prediction": prediction,
    }


def run_point(
    client: OpenAI,
    formatter: MemoryFormatter,
    items: list[dict],
    k: int,
    model: str,
    judge_model: str,
    workers: int,
) -> dict:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(
            pool.map(
                lambda it: answer_one(client, formatter, it, k, model, judge_model),
                items,
            )
        )
    rows = [r for r in rows if "error" not in r]
    ans = [r for r in rows if r["category"] in (1, 2, 3, 4)]
    adv = [r for r in rows if r["category"] == 5]

    def mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    per_cat = {}
    for cat in (1, 2, 3, 4):
        sub = [r for r in ans if r["category"] == cat]
        if sub:
            per_cat[CATEGORY_NAMES[cat]] = mean([r["judge"] for r in sub])

    return {
        "top_k": k,
        "n_answerable": len(ans),
        "n_adversarial": len(adv),
        "answerable_judge": mean([r["judge"] for r in ans]),
        "adversarial_judge": mean([r["judge"] for r in adv]),
        "answerable_refusal_rate": mean([float(r["refused"]) for r in ans]),
        "answerable_gold_present": mean([float(r["gold_present"]) for r in ans]),
        "per_category_judge": per_cat,
        "errors": len(items) - len(rows),
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(description="ITM abstention operating curve on LOCOMO")
    p.add_argument("--data-file", required=True)
    p.add_argument("--model", default="gpt-4.1-nano")
    p.add_argument("--judge-model", default="gpt-4.1-nano")
    p.add_argument("--answerable", type=int, default=250)
    p.add_argument("--adversarial", type=int, default=150)
    p.add_argument("--full", action="store_true", help="use every question")
    p.add_argument("--sweep", default=",".join(str(k) for k in DEFAULT_SWEEP))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out-dir", default="benchmarks/results")
    p.add_argument(
        "--baseline-gate",
        action="store_true",
        help="run with the chat-tuned gate (A0B0) instead of transcript ingest",
    )
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    sweep = sorted({int(x) for x in args.sweep.split(",")})
    max_k = max(sweep)

    samples = json.load(open(args.data_file))
    if args.full:
        selected = {
            si: list(range(len(s.get("qa", [])))) for si, s in enumerate(samples)
        }
    else:
        selected = select_questions(
            samples, args.answerable, args.adversarial, args.seed
        )

    kwargs = _base_kwargs()
    kwargs["top_k"] = max_k
    if not args.baseline_gate:
        kwargs["transcript_ingest"] = True
        kwargs["spreading_debias_enabled"] = True
    config = MemoryConfig(**kwargs)

    tag = "A0B0" if args.baseline_gate else "A1B1"
    print(f"Config {tag}: sweep={sweep} max_k={max_k}")
    print("Loading BGE-M3...")
    embedder = CachedEmbedder(config, Path(args.out_dir) / "_embed_cache.pkl")
    formatter = MemoryFormatter()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=8)

    items: list[dict] = []
    t0 = time.time()
    for si in sorted(selected):
        sample = samples[si]
        layer = MemoryLayer(config, embedder)
        turns = _collect_turns(sample["conversation"])
        text_to_dia = {t["text"]: t["dia_id"] for t in turns}
        # ingest_conversation embeds turn-by-turn; batch them into the cache
        # first so those calls are all hits.
        embedder.embed_batch([t["text"] for t in turns])
        ingest_conversation(layer, embedder, sample["conversation"])
        items.extend(
            retrieve_all(layer, embedder, sample, selected[si], max_k, text_to_dia)
        )
        embedder.flush()
        print(
            f"  [{si + 1}/{len(samples)}] {sample.get('sample_id')}: "
            f"{len(layer.memories)} memories, {len(selected[si])} questions "
            f"({time.time() - t0:.0f}s)"
        )

    n_ans = sum(1 for i in items if i["category"] in (1, 2, 3, 4))
    n_adv = sum(1 for i in items if i["category"] == 5)
    print(
        f"\nRetrieved {len(items)} questions ({n_ans} answerable, {n_adv} adversarial)"
    )
    print(f"Model calls ahead: ~{len(items) * len(sweep) * 2}\n")

    points = []
    for k in sweep:
        t1 = time.time()
        pt = run_point(
            client, formatter, items, k, args.model, args.judge_model, args.workers
        )
        points.append(pt)
        print(
            f"  k={k:<3} answerable={pt['answerable_judge']:.4f} "
            f"adversarial={pt['adversarial_judge']:.4f} "
            f"refusal={pt['answerable_refusal_rate']:.4f} "
            f"gold@k={pt['answerable_gold_present']:.4f} "
            f"({time.time() - t1:.0f}s)"
        )

    out = {
        "benchmark": "LOCOMO (ACL 2024)",
        "config": tag,
        "answer_model": args.model,
        "judge_model": args.judge_model,
        "embedding_model": config.embedding_model,
        "sweep_knob": "top_k (retrieval budget)",
        "sweep": sweep,
        "seed": args.seed,
        "subsample": None if args.full else {"answerable": n_ans, "adversarial": n_adv},
        "note": (
            "k=0 is the empty-context anchor, i.e. the always-refuse policy. Any "
            "adversarial accuracy must be read against it. Adversarial questions "
            "are scored by refusal detection, answerable by LLM judge."
        ),
        "points": points,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"operating_curve_{tag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\nWrote {path}")
    print(
        f"\n{'k':>4} {'answerable':>11} {'adversarial':>12} {'refusal':>9} {'gold@k':>8}"
    )
    for pt in points:
        print(
            f"{pt['top_k']:>4} {pt['answerable_judge']:>11.4f} "
            f"{pt['adversarial_judge']:>12.4f} {pt['answerable_refusal_rate']:>9.4f} "
            f"{pt['answerable_gold_present']:>8.4f}"
        )


if __name__ == "__main__":
    main()

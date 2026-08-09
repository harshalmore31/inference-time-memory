"""Retrieval baselines for ITM on LOCOMO -- the apples-to-apples comparison.

The three-arm ablation in `evaluate_memory.py` compares ITM against itself with
two flags flipped. It answers "do these two behaviors help?" but not "does the
recall machinery beat trivial retrieval?". This harness answers the second
question with everything else held fixed: same turn corpus, same BGE-M3
embeddings, same questions, same top_k.

Metric is recall@k -- whether a gold evidence turn appears in the retrieved
top-k -- so it isolates retrieval from the answer model entirely and needs
ZERO API calls.

Baselines:
  dense   cosine(embed(question), embed(turn))
  bm25    classic BM25 over turn text
  hybrid  minmax(dense) + w * minmax(bm25), w = config.recall_w_sparse

If a per-question ITM result file is passed via --itm-results, each baseline is
additionally compared to ITM on the same questions with an exact two-sided
McNemar test.

Usage:
    python benchmarks/retrieval_baseline.py \
        --data-file benchmarks/locomo/data/locomo10.json \
        --itm-results benchmarks/results/locomo_memory_results_A1B1.json
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.evaluate_memory import _collect_turns  # noqa: E402
from itm.config import MemoryConfig  # noqa: E402
from itm.embeddings import EmbeddingService  # noqa: E402

ANSWERABLE = (1, 2, 3, 4)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Robertson/Sparck-Jones BM25 over the turn corpus (Eq 19's reference point)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.n = len(self.docs)
        self.lengths = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if self.n else 0.0
        self.tf = [Counter(d) for d in self.docs]

        df: Counter = Counter()
        for doc in self.docs:
            for term in set(doc):
                df[term] += 1
        self.idf = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, counts in enumerate(self.tf):
            for term in counts:
                self.postings[term].append(i)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                f = self.tf[i][term]
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.lengths[i] / self.avgdl
                )
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def mcnemar(a: list[int], b: list[int]) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired binary outcomes. Returns (n01, n10, p)."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return n01, n10, min(1.0, p)


def parse_args():
    p = argparse.ArgumentParser(description="LOCOMO retrieval baselines for ITM")
    p.add_argument("--data-file", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--itm-results",
        default=None,
        help="per-question ITM result JSON, for the paired McNemar comparison",
    )
    p.add_argument("--out-dir", default="benchmarks/results")
    return p.parse_args()


def main():
    args = parse_args()
    config = MemoryConfig()
    w_sparse = config.recall_w_sparse
    samples = json.load(open(args.data_file))

    itm_map = {}
    if args.itm_results:
        for row in json.load(open(args.itm_results)):
            itm_map[(row["sample_id"], row["question"])] = row

    print(f"Loading {config.embedding_model}...")
    embedder = EmbeddingService(config)

    names = ["dense", "bm25", "hybrid"]
    paired: dict[str, list[int]] = {n: [] for n in names}
    paired["itm"] = []
    cats: list[int] = []
    unmatched = 0

    for si, sample in enumerate(samples):
        sid = sample.get("sample_id")
        turns = _collect_turns(sample["conversation"])
        texts = [t["text"] for t in turns]
        dias = [t["dia_id"] for t in turns]
        print(f"  [{si + 1}/{len(samples)}] {sid}: {len(texts)} turns", flush=True)

        emb = np.vstack([np.asarray(v) for v in embedder.embed_batch(texts)])
        bm25 = BM25(texts)

        qas = [
            qa
            for qa in sample.get("qa", [])
            if qa.get("category") in ANSWERABLE and (qa.get("evidence") or [])
        ]
        if not qas:
            continue
        q_emb = np.vstack(
            [np.asarray(v) for v in embedder.embed_batch([q["question"] for q in qas])]
        )

        for qi, qa in enumerate(qas):
            if itm_map:
                row = itm_map.get((sid, qa["question"]))
                if row is None:
                    unmatched += 1
                    continue
                paired["itm"].append(1 if row["gold_present"] else 0)

            gold = set(qa["evidence"])
            dense = emb @ q_emb[qi]
            lex = bm25.scores(qa["question"])
            scores = {
                "dense": dense,
                "bm25": lex,
                "hybrid": minmax(dense) + w_sparse * minmax(lex),
            }
            for name in names:
                top = np.argsort(-scores[name])[: args.top_k]
                paired[name].append(1 if any(dias[j] in gold for j in top) else 0)
            cats.append(qa["category"])

    n = len(cats)
    report = {
        "benchmark": "LOCOMO (ACL 2024)",
        "metric": f"recall@{args.top_k} (gold evidence turn in top-{args.top_k})",
        "embedding_model": config.embedding_model,
        "hybrid_weight": w_sparse,
        "n_questions": n,
        "unmatched": unmatched,
        "note": (
            "Same corpus, same embeddings, same questions, same top_k as the "
            "transcript-ingest arm. No API calls: this measures retrieval only."
        ),
        f"recall_at_{args.top_k}": {
            k: round(sum(v) / n, 4) for k, v in paired.items() if v
        },
        "per_category": {},
        "mcnemar_vs_itm": {},
    }
    for cat in sorted(set(cats)):
        idx = [i for i, c in enumerate(cats) if c == cat]
        report["per_category"][str(cat)] = {
            "n": len(idx),
            **{
                k: round(sum(paired[k][i] for i in idx) / len(idx), 4)
                for k, v in paired.items()
                if v
            },
        }
    if paired["itm"]:
        for name in names:
            n01, n10, p = mcnemar(paired["itm"], paired[name])
            report["mcnemar_vs_itm"][name] = {
                "itm_only_wins": n10,
                "baseline_only_wins": n01,
                "delta_recall_pp": round(
                    100 * (sum(paired[name]) - sum(paired["itm"])) / n, 2
                ),
                "p_value_exact_two_sided": round(p, 6),
                "significant_at_0.05": bool(p < 0.05),
            }

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "retrieval_baseline.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nn = {n} answerable questions with annotated evidence")
    for k, v in report[f"recall_at_{args.top_k}"].items():
        print(f"  {k:<8} recall@{args.top_k} = {v:.4f}")
    if report["mcnemar_vs_itm"]:
        print("\nPaired exact McNemar vs ITM:")
        for name, m in report["mcnemar_vs_itm"].items():
            sig = "significant" if m["significant_at_0.05"] else "not significant"
            print(
                f"  {name:<8} {m['delta_recall_pp']:+.2f}pp  p={m['p_value_exact_two_sided']:.4f}  ({sig})"
            )
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

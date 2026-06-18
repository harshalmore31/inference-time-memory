# Codebase Audit: Inference-Time Memory (ITM)

## Archetype match

This is an **ai-native-team / research-library** project, not a full-stack app. It is a pure Python memory library: 29 NN-derived equations spread across a `itm/` package (config, embeddings, graph, hierarchy, storage, formatting, stats, patch) plus a thin `gpt.py` chat loop that monkey-patches the OpenAI client via `enable_memory`. There is **no frontend stack and that is correct** — the default Next.js + Tailwind + Postgres archetype does not apply to a library, so its absence is not drift.

The architecture is sound and buildable. The package imports cleanly with no circular-import failure (the `graph.py` -> `core.py` back-edge is correctly broken with a `TYPE_CHECKING` guard), the domain split is sensible, cosine-similarity math is centralized as reused static helpers on `EmbeddingService` rather than duplicated, and the additive flag-gated pattern (every Phase 2/3/4 equation attaches via a config flag + a focused private helper) means new equations land without destabilizing existing code. The whole suite is green: **179 tests pass in 0.14s** on a deterministic `FakeEmbedder`, no network or model download required.

Where it drifts hard is the owner's stated Python conventions and the archetype's "code reviewed before accepted" guardrail:

- **No packaging**: no `pyproject.toml`, no `src/` layout, no `setup.py`. The owner convention mandates `src/ + pyproject.toml`; this repo has a flat `itm/` package and a 4-line unpinned `requirements.txt` — the exact artifact the convention says NOT to rely on.
- **No formatter/linter/type-checker wired up**: no `ruff`/`black`/`mypy` config, none installed in the venv (only `pytest`). 55 source lines already exceed black's 88-char limit, confirming nothing enforces style.
- **No CI and no automation gate**: `.github/` is absent; nothing runs tests, lint, or type-checks before code is accepted. `pytest` itself is not even in `requirements.txt`, so a fresh clone cannot run the suite.
- **File-size cap broken**: `core.py` is 1286 lines (3.2x the 200-400 cap), `hierarchy.py` 435, `graph.py` 405. No `utils.py` exists; per-equation helpers live inside the large `MemoryLayer` class.

Secrets hygiene is the one thing fully right: `.env` is gitignored (`git check-ignore .env` exits 0), and the only key read is `OPENAI_API_KEY` via `os.environ`, no hardcoded fallback.

## Recent activity

Only **2 commits exist ever** (`7ca528c v.1`, `7592024 updates!`), and the working tree is heavily in flight. The **entire Phase 2/3/4 implementation is uncommitted**: 12 modified files (+2001 / -319) plus 6 untracked new source/test files — `itm/embeddings_api.py`, `itm/hierarchy.py`, `test_gpt.py`, `tests/test_phase2.py`, `tests/test_phase3.py`, `tests/test_phase4.py`. This is a large unreviewed surface (2001 insertions) sitting on a 2-commit history with no CI gate. The guardrails this audit recommends should be in place **before** this diff lands, not after. No blockers prevent committing it, but it should be treated as one reviewable unit.

## Blockers (must fix before new work)

None. Every claimed blocker was verified factually true but downgraded on impact: the suite is green, the public surface (`MemoryLayer`, `update`, `recall*`, `enable_memory`) is stable and exercised by 7 test files, and nothing here makes authoring the next equation unsafe or wrong. The findings below are real debt to fix soon, but none of them gates new code.

## Warnings (should fix soon)

1. **`core.py` god-module (1286 lines, ~1147-line `MemoryLayer` class, ~35 methods).** 3.2x the 200-400 line cap. The per-equation math is already factored into focused private helpers (`_prospect_strength_delta`, `_multichannel_contradiction`, `_soft_decision`, `_cold_start_factor`, `_compute_multi_head_activations`), so `update()` is a readable orchestrator, not inline math — but the file should be split by concern into importable modules (recall/scoring, gating/contradiction, strength/decay math) leaving `core.py` a coordinator under 400 lines. `itm/core.py:139-1286`.

2. **`update()` is a single ~159-line method** (`itm/core.py:1026-1184`) covering decay, context, cold start, create-vs-strengthen, contradiction, soft drift, and the full gate path. All four outcomes (strengthened/created/contradiction/filtered) are asserted in tests, so it is tested through the public seam, but decompose into `_handle_strengthen` / `_handle_contradiction` / `_handle_gated_create` so it reads as ~20 lines of orchestration.

3. **`hierarchy.py` (435) and `graph.py` (405) also exceed the 400-line cap.** Extract the L2/L3 centroid helpers out of `MemoryHierarchy` and split graph spreading from edge construction.

4. **No formatter/linter/type-checker.** No `pyproject.toml`, no `ruff`/`black`/`mypy` config, none installed. 55 lines over 88 chars. Add `pyproject.toml` with `[tool.black]`/`[tool.ruff]`/`[tool.mypy]`, declare them as dev deps, install, and run `black .` + `ruff check .`.

5. **No CI / automation gate; `pytest` not in `requirements.txt`.** A fresh clone cannot run the suite even though all 8 test files are pytest-style. Add `pytest` (and pin deps), then a minimal workflow running `pytest -q`, ruff, black --check, mypy on push/PR.

6. **Zero error-path tests anywhere.** No `pytest.raises` / `assertRaises` / `except` across all test files (verified: 0 matches). The only two raise sites — missing-key `ValueError` at `embeddings_api.py:89` and `:151` — are never asserted, and the `results = [None] * len(texts)` batch pattern is untested. Add error-path tests for missing-key, dimension-mismatch on load, and empty-input embed.

7. **`patch.py` and `embeddings_api.py` have no unit tests.** `enable_memory` is the public integration surface used by `gpt.py`, yet `grep 'patch|enable_memory' tests/*.py` returns nothing. `test_gpt.py` exercises `MemoryLayer` directly, never the patch wrapper. The recall/inject/background-update lifecycle, list[dict] input parsing, and empty-recall short-circuit are all unverified. Mock the OpenAI client and test the wrapper; no live API needed.

8. **`disable_memory()` is broken dead code** (`itm/patch.py:203-206`). It restores only `if hasattr(client.responses.create, "__wrapped__")`, but `patched_create` is never decorated with `functools.wraps`, so `__wrapped__` is never set — the function is a guaranteed silent no-op that leaves the patch active. It is exported and documented as public API but has zero callers. Stash the original on `MemoryState` and restore from that.

9. **Equation 24 (multi-channel contradiction) formula-vs-code mismatch + dead code** (`itm/core.py:704-718`). The documented `sim_mc = w_t*topic + w_d*disp + w_c*ctx` is computed into `agreement` and then **discarded**; the actual `return` ignores all three weights and the entire context channel (`ctx_sim`). Either drive the decision from `agreement` or rewrite the docstring/config to match the real `topic AND (val OR disp)` check and delete the unused line.

10. **Equation 2 hard-contradiction value update does not renormalize** (`itm/core.py:1103-1104`), while the Equation 2b soft-drift path on the very next branch does (`1120-1122`). Averaging two unit vectors leaves norm ~0.707, so a contradicted value under-contributes in `recall_weighted_value` (which sums raw values) — a silent bias proportional to absorbed contradictions. Renormalize after the update exactly as the soft-drift branch does.

11. **Displacement-edge weighting contradicts its own spec** (`itm/graph.py:121-123` vs `config.py:142`). The config comment claims `alpha_k`/`alpha_v` are rescaled when `displacement_edges_enabled`, but the code just adds `alpha_d * disp_sim` on top, pushing max semantic weight to 1.3 and admitting ~30% more edges past the fixed `theta_edge`. Either renormalize the three weights or delete the comment and retune `theta_edge`.

12. **`embeddings_api.py` imports undeclared deps.** `import cohere` (line 147) and `trust_remote_code=True` Nomic load (line 39, needs einops) — neither `cohere` nor `einops` is in `requirements.txt`. Imports are lazy (inside `__init__`) and the backends are opt-in (only `test_gpt.py` behind CLI flags), so `import itm.embeddings_api` never fails and the default BGE-M3 path is unaffected — but a fresh `pip install -r requirements.txt` then `--cohere` raises ImportError. Add them as optional extras once `pyproject.toml` exists.

13. **`MemoryStorage.load` has zero error handling** (`itm/storage.py:114-200`). `np.load`, `json.load`, and indexed npz field access are unguarded; a truncated `memory.npz` or malformed JSON propagates through `MemoryState.__init__` and crashes `enable_memory` on startup. The corruption mode is realistic: saves run in a background thread with non-atomic writes and only flush on the explicit quit path, so a Ctrl-C mid-save leaves truncated files that brick every subsequent launch. Wrap in try/except and fall back to fresh empty state.

14. **`embeddings_api.py` is an unwired orphan + DRY violation.** It subclasses `EmbeddingService` but `MemoryState` hardcodes `EmbeddingService(self.config)` (`patch.py:39`) and `enable_memory` takes no embedder argument, so the production path cannot select a backend. Its three `embed_batch` implementations are near-identical cache loops. Either thread an optional embedder through `enable_memory` and extract a shared `_batch_with_cache`, or remove it per "start minimal."

15. **`MemoryLayer.recall()` is dead production code** (`core.py:817-845`). `patch.py` only calls `recall_graph` / `recall_with_tension`; `recall()` is referenced solely in tests. Either delete it and migrate its tests, or document it as an explicit ablation baseline.

16. **`trust_remote_code=True` is a latent RCE surface** (`embeddings_api.py:38-40`). It executes arbitrary downloaded HF code at load time. Today `embedding_model` is only ever a hardcoded constant (`config.py:9` default `BAAI/bge-m3` or test literals) with no untrusted-input path, and the default `EmbeddingService` does not set the flag — so it is not exploitable now. Pin to an exact `revision=<sha>`, document the implication, and keep `embedding_model` a trusted constant.

17. **Background-update failures are silently swallowed** (`patch.py:103-127`). `_background_update` uses try/finally with no except; a `save()` failure (disk full/permissions) is absorbed by the executor future, so the user believes memory persisted when it did not. Catch/log and record a persist-failure flag.

18. **`user_id` is used directly as a filesystem path** (`storage.py:17`, `Path(memory_dir) / user_id`) with no sanitization — a path-traversal sink that is low-risk while config-controlled but dangerous if `user_id` ever comes from request input. Allowlist `[A-Za-z0-9_-]` and assert the resolved path stays under the base.

19. **Doc/code drift.** `MEMORY.md` says `consolidate_every=10` but `config.py:182` sets 20. An undocumented "Equation 2b" (soft value drift) exists outside the canonical 29-equation registry the project's thesis tracks. Multi-head weights sum to 1.15 when the sparse head is enabled (`recall_w_sparse=0.15` added without renormalizing), violating the documented convex combination (ranking unaffected). L3 identity clustering omits the avg-pairwise-similarity gate that the structurally-identical L2 routine enforces (`hierarchy.py:260-275` vs `149-158`). `test_gpt.py` uses U+26A0 and check/cross dingbats, violating the no-emoji rule. Several test files exceed 400 lines (`test_gpt.py` 729, `test_phase2.py` 690).

## Recommendations

- Add `pyproject.toml` now with `[tool.ruff]`/`[tool.black]`/`[tool.mypy]` and a dev-dependency group (pytest, pytest-cov, ruff, black, mypy). The `src/` layout is optional given the flat package already works, but the tool config and a `pip install -e .` install (so `from itm.x` does not depend on cwd) are the real gaps.
- Pin the 4 runtime deps (`openai`, `numpy`, `sentence-transformers`, `python-dotenv`) — the monkey-patch and the npz schema are both sensitive to major-version bumps, and unpinned deps make the build non-reproducible.
- Add a CI workflow running `pytest -q`, ruff, black --check, and mypy, with a coverage floor (`pytest --cov=memory`), so the line caps and 88-char limit stop drifting silently.
- Commit the Phase 2/3/4 work as one reviewable unit behind the new CI gate, rather than leaving +2001 lines uncommitted.
- Add property tests asserting the documented formulas for Eq 2 (unit-norm value after contradiction), Eq 24 (weighted sim drives the decision), and Eq 4+18 (max edge-weight bound) — each divergence above would have been caught by a single per-equation property test.
- Record the `core.py` god-module split and the storage-serialization-ownership change as XIPs before refactoring (plan-before-code), since they touch every module importing `core`.
- Move the pure data types (`MemoryEntry`, `RecallResult`, `SparseIndex`) into a small `types` module so `formatting`/`graph`/`storage` do not drag in the 1286-line `core` to use a dataclass.

## Next phase

**implementation.** The foundation is structurally sound — the package imports cleanly, module boundaries are sensible, the math is faithful with strong numerical hygiene, and 179 tests pass green — so this needs guardrails, targeted bug fixes, and a refactor of the oversized `core.py`, not a ground-up architecture redesign.

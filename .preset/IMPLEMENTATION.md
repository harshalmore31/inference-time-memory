# Implementation: Packaging, Tooling, CI, and Docs Baseline

Phase scope: packaging, tooling, CI, docs, formatting, and emoji removal only.
No program logic, equation math, or file structure was changed. `black` and
`ruff --fix` (safe autofixes: import sorting, unused-import removal, f-string and
PEP-604 cleanups) were applied; these do not alter behavior.

## Modules built

New files:

- `pyproject.toml` — PEP 621 project metadata. Distribution name
  `inference-time-memory`, import name `itm`, `requires-python = ">=3.12"`.
  Floored runtime deps (`openai>=2.21`, `numpy>=1.26`, `sentence-transformers>=5.0`,
  `python-dotenv>=1.0`); `sentence-transformers` kept as a core dep so the default
  BGE-M3 embedder works out of the box. Optional extras `api`
  (`cohere>=5.0`, `einops>=0.8`) and `dev` (`pytest>=8`, `ruff>=0.6`, `black>=24`,
  `mypy>=1.11`). `[tool.setuptools] packages = ["itm"]` packages only the
  flat `itm/` package (gpt.py / test_gpt.py stay top-level scripts).
  Tool config: `[tool.black]` (line-length 88, py312), `[tool.ruff]` +
  `[tool.ruff.lint]` (line-length 88; select E,F,I,UP,B with pragmatic ignores),
  `[tool.mypy]` (lenient: ignore_missing_imports, no strict mode).
- `LICENSE` — standard MIT text, "Copyright (c) 2026 Harshal More".
- `.github/workflows/ci.yml` — GitHub Actions on push + pull_request to `main`.
  ubuntu-latest, actions/checkout@v4, actions/setup-python@v5 (3.12),
  `pip install -e .[dev]`, then `ruff check .`, `black --check .`,
  `mypy itm` (non-blocking, continue-on-error), `pytest -q tests`. test_gpt.py
  is excluded with an inline comment (multi-GB model download + network).
- `itmips/itmip-01-adopt-xip-methodology.md` — Status Accepted. Adopt XIP/ADR
  decision records (ITMIP prefix) in `itmips/`.
- `itmips/itmip-02-clean-slate-history-rewrite.md` — Status Implemented.
  Records the zero-risk single-root history rewrite (original 2 commits embedded
  a 3.6MB PDF + personal runtime memory data; never pushed). Backup at branch
  `backup/messy-history` + tag `pre-reset-backup`.
- `itmips/itmip-03-packaging-and-tooling-baseline.md` — Status Implemented.
  Records the flat `itm/` layout decision (vs src/), floored deps, ruff/black/
  mypy config, and the CI gate, with alternatives weighed.

Changed files:

- `requirements.txt` — reduced to `-e .` (pyproject is the single source of truth
  for dependencies).
- `README.md` — light accuracy pass: 21->29 equations and 149->179 tests
  throughout; Installation section rewritten to `pip install -e .` /
  `.[dev]` / `.[api]`; added a License section; project-structure block updated
  (added hierarchy.py, embeddings_api.py, test_gpt.py; corrected test count to
  179 across 7 files); replaced the broken `research.md` link (research/ is
  gitignored, absent in the public repo) with inline prose. No wholesale rewrite.
- `test_gpt.py` — replaced the three emoji dingbats in `check()` output
  (U+2713 check, U+26A0 warning, U+2717 cross) with ASCII tokens
  `[PASS]` / `[WARN]` / `[FAIL]`. ANSI color codes retained (not emojis).
- `gpt.py`, `itm/__init__.py`, `itm/config.py`, `itm/core.py`,
  `itm/embeddings_api.py`, `itm/formatting.py`, `itm/graph.py`,
  `itm/hierarchy.py`, `itm/patch.py`, `itm/stats.py`,
  `itm/storage.py`, and all 7 files under `tests/` — `black` reformatting
  (whitespace/wrapping) + `ruff --fix` safe autofixes (import sorting, unused
  imports, f-string-without-placeholder, PEP-604 Optional). No logic changes;
  `__init__.py` re-exports and `__all__` unchanged.

## Tests

All commands run with `/Users/harshalmore31/code/llm-agent-learning/.venv/bin/python`
(Python 3.12.12). Installed versions confirmed: openai 2.21.0, numpy 2.4.2,
sentence-transformers 5.2.3 — all satisfy the declared floors.

- `pip install -e .` — succeeded.
- `python -c "import itm"` — `import itm: OK`.
- `ruff check .` — `All checks passed!`, exit 0.
- `black --check .` — `21 files would be left unchanged`, exit 0.
- `pytest -q tests` — **179 passed in 0.12s** (matches the pre-formatting
  baseline of 179; zero regressions).
- `mypy itm` — runs and reports 32 errors in 4 files (expected; the codebase
  is largely untyped). Configured as non-blocking in CI via continue-on-error.
- Emoji scan over tracked source (itm/, tests/, gpt.py, test_gpt.py,
  README.md, LICENSE, itmips/, .github/, pyproject.toml; *.py, *.md, *.yml,
  *.toml) for emoji/dingbat code-point ranges — **no emoji dingbats found**.

## Deviations

- **requirements.txt**: chose to keep the file but reduce it to `-e .` (rather
  than deleting it) so `pip install -r requirements.txt` still works and
  pyproject.toml remains the single source of truth for dependencies.
- **mypy enforcement**: included in CI but as a non-blocking step
  (`continue-on-error: true`) targeting only `itm`. The package has 32 mypy
  errors today (untyped code); a blocking gate would fail immediately and force
  out-of-scope typing work. Full typing is a later phase.
- **sentence-transformers as a core dep** (not an extra): it is the default
  BGE-M3 embedder, so an out-of-the-box `pip install -e .` must pull it in.
- **Ruff lint left unfixed (rules ignored, not code changed)**: of 53 initial
  findings, 35 were safe-autofixed. The remainder would require touching logic or
  intentional patterns, so the rules were ignored in `[tool.ruff.lint]` with
  rationale comments rather than edited:
  - `F841` (8, unused local) — includes the deliberately-deferred Eq 24 discarded
    `agreement`/`ctx_sim` dead code flagged in AUDIT warnings 9 & 19. Deleting
    risks behavior change.
  - `B007` (6, unused loop variable) — renaming to `_` touches logic lines.
  - `E402` (5, import after code) — the intentional `load_dotenv()`-before-import
    pattern in gpt.py / test_gpt.py. Reordering changes execution order.
  - `B008`, `B905` — pre-emptively ignored (intentional default-arg calls /
    zip-without-strict; logic-touching, deferred).
- **Removed `UP038` from the ignore list**: that rule was deleted in ruff 0.15
  (installed 0.15.17) and ignoring it is a no-op warning. The autofixed UP rule
  was `UP045` (PEP-604 Optional), which is behavior-neutral.
- **Non-ASCII that is NOT emoji was intentionally left in place**: the math
  notation throughout the codebase (Greek letters α/β/η/σ/Σ/θ/λ, arrows →/←,
  em-dashes —, box-drawing ─/═, subscripts) is legitimate equation/diagram
  notation consistent across all of `itm/`, not emoji. Boss's rule is
  "no emojis"; only true emoji dingbats were removed. Stripping all non-ASCII
  would be a large cosmetic change touching every source file and is out of
  scope.
- **README**: the `research.md` link was replaced because `research/` and
  `research.md` are gitignored and will not exist in the public repo — a
  guaranteed broken link otherwise. Also corrected the "Enable all 21 equations"
  code comment to "Enable the Phase 2/3 equations (Eq 12-21)" since the config
  block only toggles those flags.

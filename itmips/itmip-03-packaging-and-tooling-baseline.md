# ITMIP-03: Packaging and Tooling Baseline

---
Status: Implemented
Date: 2026-06-18
Author: Harshal More
---

## Context

The codebase audit (`.preset/AUDIT.md`) flagged the absence of any packaging,
linting, formatting, type-checking, or CI gate as the primary debt blocking a
public release:

- No `pyproject.toml`; runtime deps lived in a 4-line unpinned `requirements.txt`
  with `pytest` missing, so a fresh clone could not even run the suite.
- No `ruff` / `black` / `mypy` configuration and none installed in the venv;
  dozens of lines already exceeded the 88-char convention with nothing enforcing
  it.
- No `.github/` and no automation gate; nothing ran tests or lint before code
  was accepted.
- `embeddings_api.py` imports `cohere` and (via `trust_remote_code`) `einops`,
  neither declared anywhere.

The audit confirmed the flat `itm/` package imports cleanly with no circular
imports, and 179 tests pass on a deterministic `FakeEmbedder`.

## Decision

Establish a packaging and tooling baseline:

- **Layout:** keep the flat `itm/` package; do **not** migrate to `src/`. Add
  `pyproject.toml` (PEP 621) configured so `pip install -e .` makes `import
  memory` work independent of the current working directory. `gpt.py` and
  `test_gpt.py` remain top-level scripts, not part of the installed package
  (`[tool.setuptools] packages = ["itm"]`).
- **Distribution name:** `inference-time-memory`; import package is `itm`.
- **Runtime deps:** floored (not hard-pinned, since this is a library):
  `openai>=2.21`, `numpy>=1.26`, `sentence-transformers>=5.0`,
  `python-dotenv>=1.0`. `sentence-transformers` stays a core dependency so the
  default BGE-M3 embedder works out of the box.
- **Optional extras:** `api = ["cohere>=5.0", "einops>=0.8"]` for the alternate
  embedding backends in `embeddings_api.py`; `dev = ["pytest>=8", "ruff>=0.6",
  "black>=24", "mypy>=1.11"]`.
- **`requirements.txt`:** reduced to `-e .` so `pyproject.toml` is the single
  source of truth for dependencies.
- **Tooling config (in `pyproject.toml`):** `black` (line-length 88, py312);
  `ruff` (line-length 88; select E, F, I, UP, B with pragmatic ignores chosen so
  the build is clean without any logic changes); `mypy` lenient
  (`ignore_missing_imports = true`, no strict mode — full typing is a later
  phase).
- **CI:** GitHub Actions on push/PR to `main` (ubuntu-latest, Python 3.12) runs
  `ruff check .`, `black --check .`, and `pytest -q tests`. `mypy` runs as a
  non-blocking step (`continue-on-error: true`). `test_gpt.py` is excluded from
  CI because it downloads a multi-GB model and needs network access.

## Consequences

- A fresh clone can install (`pip install -e .[dev]`) and run the full suite.
- Style and import order are enforced and stop drifting silently.
- Alternate embedding backends are installable via the `api` extra without
  bloating the default install.
- Type coverage is intentionally not enforced yet; mypy is advisory until a
  dedicated typing phase.

## Alternatives considered

- **`src/` layout:** rejected for now. The owner convention nominally prefers
  `src/`, but the audit confirmed the flat package already imports cleanly and
  "start minimal, grow organically" applies. Moving to `src/` is a mechanical
  change that can happen later if import-shadowing problems ever appear; there is
  no evidence they do today.
- **Hard-pinning runtime deps:** rejected. As a library, pinning exact versions
  would create resolver conflicts for downstream users. Lower-bound floors keep
  the build reproducible enough while staying installable alongside other
  packages. (Applications depending on this can pin in their own lockfile.)
- **Making `sentence-transformers` an optional extra:** rejected. It is the
  default embedder; an out-of-the-box install must work without extras.
- **Enforcing mypy / strict typing in CI now:** rejected. The codebase is largely
  untyped; a blocking type gate would fail immediately and force premature,
  out-of-scope typing work.

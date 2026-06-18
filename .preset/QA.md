## Test results

**Suite:** `pytest -q tests`
**Result:** 179 passed, 0 failed, 0 errors
**Runtime:** 0.14s (wall: 0.35s)

All 179 unit tests pass using a deterministic `FakeEmbedder` — no model downloads, no network, fully offline. Lint (`ruff check .`) and format check (`black --check .`) both pass with no issues.

Fresh-venv install was attempted but not executed due to sentence-transformers pulling PyTorch (multi-GB). Static dependency analysis was used instead (see Edge cases).

---

## Edge cases

### Dependency completeness

Third-party imports found across `itm/*.py`, `gpt.py`, `tests/*.py`, `test_gpt.py`:

| Import | Package | Declared in pyproject.toml? |
|---|---|---|
| `numpy` | numpy | Yes — core dep (`numpy>=1.26`) |
| `openai` | openai | Yes — core dep (`openai>=2.21`) |
| `dotenv` | python-dotenv | Yes — core dep (`python-dotenv>=1.0`) |
| `sentence_transformers` | sentence-transformers | Yes — core dep (`sentence-transformers>=5.0`) |
| `cohere` | cohere | Yes — `api` extra (`cohere>=5.0`) |

Notes:
- `sentence_transformers` import is **lazy** (inside `EmbeddingService.__init__`, not at module level). Unit tests never trigger it because every test file's `FakeEmbedder` overrides `__init__` without calling `super()`.
- `cohere` import is **lazy** (inside `CohereEmbedding.__init__` in `embeddings_api.py`). It is correctly placed in the `api` optional extra. A user installing without `[api]` will not hit it unless they explicitly instantiate `CohereEmbedding`.
- `einops` appears in the `api` extra declaration but is **not imported anywhere** in the source tree. It is either a forward declaration for Nomic trust_remote_code internals or dead weight. Not a problem for fresh-clone correctness — only the `api` extra lists it.
- `NomicEmbedding` uses `sentence_transformers` (already a core dep) with `trust_remote_code=True`. No extra package beyond `sentence-transformers` is required at import time for Nomic.
- No undeclared third-party imports were found.

**Verdict: dependency declarations are complete and correct.**

### Fresh-clone self-sufficiency

Files a cloner receives (`git ls-files`): `.gitignore`, `.preset/AUDIT.md`, `.preset/state.json`, `README.md`, `gpt.py`, `itm/__init__.py`, `itm/config.py`, `itm/core.py`, `itm/embeddings.py`, `itm/embeddings_api.py`, `itm/formatting.py`, `itm/graph.py`, `itm/hierarchy.py`, `itm/patch.py`, `itm/stats.py`, `itm/storage.py`, `requirements.txt`, `test_gpt.py`, `tests/__init__.py`, `tests/test_advanced.py`, `tests/test_core.py`, `tests/test_gate.py`, `tests/test_graph.py`, `tests/test_phase2.py`, `tests/test_phase3.py`, `tests/test_phase4.py`.

Gitignored paths referenced in source/tests:

- `memory_data/` — referenced only as a **default config value** (`memory_dir: str = "memory_data"` in `itm/config.py:93`). It is created at runtime by `MemoryStorage._get_user_dir()`. Tests override this with pytest `tmp_path` fixtures — no test writes to the real `memory_data/`. A fresh clone has no `memory_data/` and this is fine.
- `research/`, `benchmarks/` — not referenced in any source or test file.
- `.env` — referenced only in `gpt.py` via `load_dotenv()` (optional) and in `embeddings_api.py` via `os.environ.get(...)`. The `load_dotenv()` call is gracefully no-op when `.env` is absent. Tests do not call `load_dotenv()`.

**Verdict: no source or test file hard-depends on a gitignored path. A fresh clone is self-sufficient for the unit test suite.**

### New professionalization files not yet tracked by git

The following files exist in the working tree but are **untracked** (not yet staged/committed):

- `.github/workflows/ci.yml` — the GitHub Actions CI workflow
- `LICENSE` — MIT license file
- `pyproject.toml` — build system and dependency declarations
- `itmips/` — three ITMIP decision records

These files are present locally and verified to be correct (CI workflow runs `pip install -e .[dev]` then `ruff check`, `black --check`, `mypy itm`, `pytest -q tests` in that order; `test_gpt.py` is intentionally excluded with a comment explaining why). They must be committed before the push to GitHub for the professionalization pass to take effect.

### First-run / empty-state behavior

`itm/storage.py:16-18` — `_get_user_dir()` calls `path.mkdir(parents=True, exist_ok=True)`, creating `memory_data/<user_id>/` on first access. This is called from both `save()` and `load()`.

`itm/storage.py:130` — `load()` checks `if not npz_path.exists(): return False` before attempting to read any file. A fresh machine with no saved state returns `False` without raising.

`itm/patch.py:53-58` — `MemoryState.__init__` calls `self.storage.load(self.layer)` and branches: if `loaded` is `False` (no saved state), it prints `"[Memory] Starting fresh — no saved memories found"` and proceeds normally. No exception, no crash.

Unit test `tests/test_core.py:365-370` (`test_load_nonexistent_returns_false`) explicitly asserts this behavior: `assert not storage.load(layer)` passes.

**Verdict: first-run with no saved memory is handled correctly at storage.py:18 (dir creation) and storage.py:130 (missing file guard).**

### What could not be tested

- `pip install -e .` in a fresh venv was not executed. sentence-transformers pulls PyTorch (~2 GB), which would exceed a reasonable time box. The static analysis confirms all declared deps are correct.
- `test_gpt.py` integration test was not run (requires BGE-M3 model download and `OPENAI_API_KEY`). It is intentionally excluded from CI and from this QA run.
- GitHub Actions execution was not verified (workflow file is untracked and cannot run until committed and pushed).

---

## Verdict

PASS

All 179 unit tests pass in 0.14s with zero failures. Dependency declarations are complete (no undeclared third-party imports). No test or source file hard-depends on a gitignored path. First-run empty-state is handled gracefully. Lint and format checks pass. The library is releasable once the untracked files (`.github/`, `LICENSE`, `pyproject.toml`, `itmips/`) are committed and pushed.

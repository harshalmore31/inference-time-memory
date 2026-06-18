# ITM — AI Team Build Summary

Project: Inference-Time Memory (ITM). Working folder: llm-agent-learning (folder name only).
Workflow: ai-native-team (preset). Project id: proj-fef3524674ee19f9.
Intent: hardening an existing, feature-complete research library for a public GitHub release.

## Phase recap

1. Codebase audit (0 blockers, 19 warnings). Foundation sound: clean imports, faithful math, 179 tests green. All 11 candidate blockers were adversarially verified and downgraded to debt. Next phase: implementation.
2. Implementation — repo professionalization (no logic changes):
   - pyproject.toml: flat `itm` package, MIT, floored runtime deps, `api` + `dev` extras, ruff/black/mypy config.
   - GitHub Actions CI: ruff + black --check + mypy (non-blocking) + pytest; test_gpt.py excluded (needs model + network).
   - LICENSE (MIT), README polish, requirements.txt -> `-e .`, emoji dingbats removed, black + ruff applied across the tree.
   - Decision records ITMIP-01/02/03 in itmips/.
3. Code review — APPROVE (independent Sonnet reviewer, different tier). Behavior-change check clean; all 29 equation literals verified unchanged; deferred bugs untouched.
4. QA — PASS. 179 tests green offline; dependency declarations complete; fresh-clone self-sufficient; first-run empty-state handled.
5. Deployment — pending human go for the GitHub push.

## Structural changes made this build
- History rewritten to a single clean root commit; the original 3.6MB PDF and personal runtime memory data were purged. Backup preserved at branch `backup/messy-history` + tag `pre-reset-backup` (local only).
- Package renamed `memory` -> `itm` (distinctive, collision-safe public name); all imports rewritten; gates re-verified green.
- Decision-record prefix `LAM` -> `ITM` (itmips/, ITMIP-NN); distribution name `inference-time-memory`.

## Deferred to later phases (tracked in .preset/AUDIT.md warnings 8-19)
- Equation-correctness bugs: Eq 2 contradiction renormalization, Eq 24 discarded weighted-sim dead code, displacement-edge weighting vs spec, disable_memory() no-op, storage.load() error handling, silent background-update failures.
- core.py god-module refactor (1286 lines) + extract a small types module.
- Error-path tests; unit tests for patch.py and embeddings_api.py; per-equation property tests.

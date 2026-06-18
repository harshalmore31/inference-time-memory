# ITMIP-01: Adopt XIP Decision Records

---
Status: Accepted
Date: 2026-06-18
Author: Harshal More
---

## Context

This project (`inference-time-memory`) is moving from a private prototype to a
public, GitHub-ready repository. Non-trivial architectural and process
decisions had previously lived only in commit messages and an external memory
file, with no durable, reviewable record in the repository itself. As the
codebase grows (29 equations across multiple modules, a planned `core.py`
split, and serialization-ownership changes), future contributors need to
understand why decisions were made, not just what the code currently does.

## Decision

Adopt lightweight decision records, prefixed `ITMIP` (Inference-Time-Memory
Improvement Proposal), stored in the `itmips/` folder. Each record follows the
XIP/ADR format: a title, frontmatter (`Status`, `Date`, `Author`), and the
sections Context, Decision, Consequences, and Alternatives considered. Records
are numbered sequentially and are immutable once Accepted/Implemented — a later
record supersedes an earlier one rather than editing it in place.

Statuses used: `Proposed`, `Accepted`, `Implemented`, `Superseded`, `Rejected`.

## Consequences

- Architectural decisions become discoverable and reviewable in-repo.
- Plan-before-code is enforceable: significant changes get a ITMIP first.
- A small ongoing authoring cost per non-trivial decision.

## Alternatives considered

- **No formal records (status quo):** rejected. Rationale lived only in commit
  messages and external notes, invisible to public contributors.
- **GitHub wiki / external docs:** rejected. Decouples the record from the code
  it governs; goes stale and is not reviewed alongside diffs.
- **Inline code comments only:** rejected. Comments explain local mechanics, not
  cross-cutting decisions or the alternatives that were weighed and dropped.

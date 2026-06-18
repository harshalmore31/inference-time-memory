# ITMIP-02: Clean-Slate Git History Rewrite

---
Status: Implemented
Date: 2026-06-18
Author: Harshal More
---

## Context

The repository's original history consisted of only two commits
(`7ca528c v.1`, `7592024 updates!`). Those commits permanently embedded
artifacts that should never ship in a public repository:

- A 3.6 MB research PDF (now covered by the `*.pdf` gitignore rule).
- Personal runtime memory data (`.npz` / `.json` files under `memory_data/`,
  now gitignored), i.e. real conversational facts captured during local use.

Because these were committed in early history, gitignoring them going forward
would not remove them from the repository's object store — anyone cloning the
public repo could recover the PDF and the personal data from history. The repo
had never been pushed to any remote, so its entire history was local-only.

## Decision

Rewrite history to a single clean root commit, discarding the two original
commits. Since the repo had never been pushed, this is a zero-risk operation:
no collaborator's clone is invalidated and no public object store contains the
old blobs. Before rewriting, the original history was preserved locally for
auditability:

- Branch: `backup/messy-history`
- Tag: `pre-reset-backup`

## Consequences

- The public history is clean from commit one — no embedded PDF, no personal
  memory data, no `memory_data/` blobs.
- The original two-commit history is recoverable locally via the backup branch
  and tag, should provenance ever need to be reconstructed.
- Standard caveat of any history rewrite (commit hashes change) is irrelevant
  here because nothing was ever shared.

## Alternatives considered

- **Keep history, just gitignore going forward:** rejected. The PDF and personal
  data remain permanently recoverable from history once the repo is public — a
  privacy and repo-bloat leak.
- **`git filter-repo` / BFG to surgically strip the blobs:** rejected as
  unnecessary. With only two never-pushed commits and no history worth
  preserving publicly, a clean single-root reset is simpler and leaves no
  partial-rewrite artifacts. The full original history is kept in the backup.

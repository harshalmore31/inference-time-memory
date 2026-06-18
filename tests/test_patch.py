"""Tests for the OpenAI client patch (itm/patch.py).

Covers the public off-switch (disable_memory) and background-failure surfacing.
These do NOT require the real BGE-M3 model or disk: the embedder and storage
are replaced with in-memory fakes.
"""

import numpy as np
import pytest

import itm.patch as patch_mod
from itm.config import MemoryConfig


class FakeEmbedder:
    """Deterministic in-memory embedder — no model load."""

    def __init__(self, config):
        self.config = config
        self._cache = {}

    def embed(self, text):
        if text not in self._cache:
            rng = np.random.default_rng(abs(hash(text)) % (2**31))
            v = rng.standard_normal(self.config.embedding_dim).astype(np.float32)
            self._cache[text] = v / np.linalg.norm(v)
        return self._cache[text]


class FakeStorage:
    """No-op storage — never touches disk."""

    def __init__(self, config):
        self.config = config
        self.saved = 0

    def load(self, layer):
        return False

    def save(self, layer):
        self.saved += 1
        return "fake"


class FakeResponse:
    def __init__(self, text):
        self.output_text = text


class MockResponses:
    """Stands in for client.responses; records whether create was wrapped."""

    def __init__(self):
        self.calls = 0

        def create(*args, **kwargs):
            self.calls += 1
            return FakeResponse("an answer")

        self.create = create


class MockClient:
    def __init__(self):
        self.responses = MockResponses()


@pytest.fixture
def patched_env(monkeypatch):
    """Swap in fake embedder + storage so MemoryState needs no model/disk."""
    monkeypatch.setattr(patch_mod, "EmbeddingService", FakeEmbedder)
    monkeypatch.setattr(patch_mod, "MemoryStorage", FakeStorage)
    return MemoryConfig(embedding_dim=16, embedding_model="test")


# ── FIX 6: disable_memory must actually restore the original create ──


class TestDisableMemory:
    def test_disable_restores_exact_original_create(self, patched_env):
        client = MockClient()
        original_create = client.responses.create

        state = patch_mod.enable_memory(client, patched_env)
        # After enabling, create is the wrapper (not the original).
        assert client.responses.create is not original_create

        restored = patch_mod.disable_memory(client)
        assert restored is True
        # The EXACT original callable is back.
        assert client.responses.create is original_create

    def test_call_after_disable_does_not_go_through_wrapper(self, patched_env):
        client = MockClient()

        # Sentinel: a wrapper would set _last_recalled on the state on each call.
        state = patch_mod.enable_memory(client, patched_env)
        patch_mod.disable_memory(client)

        # Reset the sentinel, then call create directly.
        state._last_recalled = None
        resp = client.responses.create(input="hello", model="x")
        assert resp.output_text == "an answer"
        # The wrapper never ran -> no recall snapshot was taken.
        assert state._last_recalled is None

    def test_disable_on_unpatched_client_is_noop(self, patched_env):
        client = MockClient()
        # Never enabled -> disable returns False, leaves create untouched.
        original = client.responses.create
        assert patch_mod.disable_memory(client) is False
        assert client.responses.create is original


# ── FIX 8: background update failures must be surfaced, not swallowed ──


class TestBackgroundFailureSurfaced:
    def test_save_failure_is_flagged_and_does_not_crash_caller(
        self, patched_env, caplog
    ):
        client = MockClient()
        state = patch_mod.enable_memory(client, patched_env)

        # Force the background save to raise (disk full / permissions analog).
        def boom(layer):
            raise OSError("disk full")

        state.storage.save = boom

        # The user-facing call must still return normally.
        with caplog.at_level("ERROR", logger="itm.patch"):
            resp = client.responses.create(input="remember this", model="x")
            assert resp.output_text == "an answer"
            # Wait for the background worker to finish.
            state.flush(timeout=5.0)

        # The failure is surfaced: flagged on the state AND logged.
        assert state._last_persist_error is not None
        assert isinstance(state._last_persist_error, OSError)
        assert any(
            "Background memory update failed" in r.message for r in caplog.records
        )

        state.shutdown()

    def test_successful_update_leaves_no_persist_error(self, patched_env):
        client = MockClient()
        state = patch_mod.enable_memory(client, patched_env)

        client.responses.create(input="remember this", model="x")
        state.flush(timeout=5.0)

        assert state._last_persist_error is None
        assert state.storage.saved >= 1
        state.shutdown()

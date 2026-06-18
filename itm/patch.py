"""Transparent memory integration for OpenAI clients.

Usage:
    from openai import OpenAI
    from itm.patch import enable_memory

    client = OpenAI()
    mem = enable_memory(client)  # That's it. Memory is now active.

    response = client.responses.create(model="gpt-4.1-nano", input="hello")
    # Response returned immediately.
    # Memory update (embed output, update, save) runs in background thread.

Performance:
  - Recall: embed query + numpy recall (must complete before API call)
  - API call: sync, as normal
  - Post-response: embed output + update + save fires in background thread
  - User sees response without waiting for memory processing
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from itm.config import MemoryConfig
from itm.core import MemoryLayer
from itm.embeddings import EmbeddingService
from itm.formatting import MemoryFormatter
from itm.stats import MemoryStats
from itm.storage import MemoryStorage


class MemoryState:
    """Holds all memory components for a patched client."""

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        self.embedder = EmbeddingService(self.config)
        self.layer = MemoryLayer(self.config, self.embedder)
        self.storage = MemoryStorage(self.config)
        self.formatter = MemoryFormatter()
        self.stats = MemoryStats()
        self._last_action: dict | None = None
        self._last_recalled: list | None = None  # Eq 16: for feedback loop

        # Background thread pool for post-response processing
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory")
        self._lock = threading.Lock()
        self._pending: list[threading.Event] = []

        # Load existing memories from disk
        loaded = self.storage.load(self.layer)
        if loaded:
            n = len(self.layer.memories)
            print(f"[Memory] Loaded {n} memories (timestep {self.layer.timestep})")
        else:
            print("[Memory] Starting fresh — no saved memories found")

    @property
    def last_action(self) -> dict | None:
        return self._last_action

    def flush(self, timeout: float | None = None):
        """Wait for all background memory updates to complete.

        Call this before reading last_action or before exit.
        """
        with self._lock:
            events = list(self._pending)

        for event in events:
            event.wait(timeout=timeout)

    def shutdown(self):
        """Flush pending work and shut down the thread pool."""
        self.flush()
        self._executor.shutdown(wait=True)


def _extract_input(kwargs: dict) -> str:
    """Pull user input text from kwargs, handling str and list[dict] formats."""
    user_input = kwargs.get("input", "")
    if isinstance(user_input, list):
        text_parts = []
        for item in user_input:
            if isinstance(item, dict) and "content" in item:
                text_parts.append(str(item["content"]))
            elif isinstance(item, str):
                text_parts.append(item)
        user_input = " ".join(text_parts)
    return str(user_input)


def _inject_context(kwargs: dict, memory_context: str):
    """Prepend memory context to instructions."""
    original = kwargs.get("instructions", "")
    kwargs["instructions"] = (
        f"{original}\n\n{memory_context}" if original else memory_context
    )


def _background_update(
    state: MemoryState, user_input: str, output_text: str, done: threading.Event
):
    """Background worker: embed output, update memory, save to disk, apply feedback.

    Runs in thread pool after the response has been returned to the user.
    The input embedding is already cached from the recall phase — free lookup.
    """
    try:
        input_emb = state.embedder.embed(user_input)  # cached from recall — instant
        output_emb = state.embedder.embed(output_text)  # new — runs model inference

        with state._lock:
            state._last_action = state.layer.update(
                input_emb, output_emb, user_input, output_text
            )

            # Equation 16: Feedback loop — strengthen/weaken based on response
            if state.config.feedback_enabled and state._last_recalled:
                state.layer.apply_feedback(output_emb, state._last_recalled)

            state.storage.save(state.layer)
    finally:
        done.set()
        with state._lock:
            if done in state._pending:
                state._pending.remove(done)


def enable_memory(client, config: MemoryConfig | None = None) -> MemoryState:
    """Enable transparent memory on an OpenAI client.

    Patches client.responses.create so every call:
      1. Recalls relevant memories (sync — must complete before API call)
      2. Injects them into instructions
      3. Forwards to the real API (sync)
      4. Fires background thread: embed output + update memory + save

    Args:
        client: openai.OpenAI instance.
        config: Optional MemoryConfig.

    Returns:
        MemoryState for inspecting/controlling memory.
    """
    state = MemoryState(config)

    original_create = client.responses.create

    def patched_create(*args, **kwargs):
        user_input = _extract_input(kwargs)

        # ── Recall (sync, must complete before API call) ──
        recalled_list = []
        if state.layer.memories and user_input.strip():
            query_emb = state.embedder.embed(user_input)

            if state.config.tension_enabled:
                # Equation 13: tension-aware recall
                recall_result = state.layer.recall_with_tension(
                    query_emb,
                    query_text=user_input,
                )
                recalled_list = recall_result.all_results
                if recall_result.all_results:
                    _inject_context(
                        kwargs, state.formatter.format_with_tension(recall_result)
                    )
            elif state.config.channels_enabled:
                # Equation 14: channel-aware formatting
                recalled_list = state.layer.recall_graph(
                    query_emb,
                    query_text=user_input,
                )
                if recalled_list:
                    _inject_context(
                        kwargs, state.formatter.format_channeled(recalled_list)
                    )
            else:
                recalled_list = state.layer.recall_graph(
                    query_emb,
                    query_text=user_input,
                )
                if recalled_list:
                    _inject_context(
                        kwargs, state.formatter.format_for_prompt(recalled_list)
                    )

        # Snapshot recalled for feedback loop (Eq 16)
        with state._lock:
            state._last_recalled = list(recalled_list)

        # ── Forward to real API (sync) ──
        response = original_create(*args, **kwargs)

        # ── Background: embed output + update + save + feedback ──
        output_text = getattr(response, "output_text", "") or ""
        if user_input.strip() and output_text.strip():
            done = threading.Event()
            with state._lock:
                state._pending.append(done)
            state._executor.submit(
                _background_update, state, user_input, output_text, done
            )

        return response

    client.responses.create = patched_create
    return state


def disable_memory(client):
    """Remove the memory patch, restoring original behavior."""
    if hasattr(client.responses.create, "__wrapped__"):
        client.responses.create = client.responses.create.__wrapped__

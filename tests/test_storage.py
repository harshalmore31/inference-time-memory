"""Tests for itm/storage.py persistence — corruption resilience + atomicity.

FIX 7: load() must not crash on a truncated .npz or malformed JSON; it falls
back to a fresh empty state. Saves are atomic (temp file + rename).
"""

import json

import numpy as np

from itm.config import MemoryConfig
from itm.core import MemoryLayer
from itm.storage import MemoryStorage


def random_embedding(dim, rng):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    def __init__(self, config):
        self.config = config

    def embed(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        return random_embedding(self.config.embedding_dim, rng)


def _make_layer(tmp_path, seed=0, n=3):
    cfg = MemoryConfig(
        embedding_dim=16, embedding_model="test", memory_dir=str(tmp_path)
    )
    layer = MemoryLayer(cfg, FakeEmbedder(cfg))
    rng = np.random.default_rng(seed)
    for i in range(n):
        k = random_embedding(16, rng)
        v = random_embedding(16, np.random.default_rng(seed + 100 + i))
        layer._create_memory(k, v, f"input {i}", f"output {i}")
    layer.timestep = 7
    return cfg, layer


def _user_dir(cfg):
    from pathlib import Path

    return Path(cfg.memory_dir) / cfg.user_id


def test_roundtrip_then_corrupt_npz_falls_back_to_fresh(tmp_path):
    cfg, layer = _make_layer(tmp_path)
    storage = MemoryStorage(cfg)
    storage.save(layer)

    # Sanity: a clean load succeeds.
    fresh = MemoryLayer(cfg, FakeEmbedder(cfg))
    assert storage.load(fresh) is True
    assert len(fresh.memories) == 3

    # Truncate the npz to simulate a Ctrl-C mid-save / disk truncation.
    npz_path = _user_dir(cfg) / "memory.npz"
    data = npz_path.read_bytes()
    npz_path.write_bytes(data[: len(data) // 2])

    corrupt = MemoryLayer(cfg, FakeEmbedder(cfg))
    # Must NOT raise; returns False and resets to empty state.
    result = storage.load(corrupt)
    assert result is False
    assert corrupt.memories == []
    assert corrupt.timestep == 0
    assert corrupt.context_vector is None


def test_garbage_npz_falls_back_to_fresh(tmp_path):
    cfg, layer = _make_layer(tmp_path)
    storage = MemoryStorage(cfg)
    storage.save(layer)

    npz_path = _user_dir(cfg) / "memory.npz"
    npz_path.write_bytes(b"this is not a valid npz file at all")

    corrupt = MemoryLayer(cfg, FakeEmbedder(cfg))
    assert storage.load(corrupt) is False
    assert corrupt.memories == []


def test_malformed_json_sidecar_falls_back_to_fresh(tmp_path):
    cfg, layer = _make_layer(tmp_path)
    storage = MemoryStorage(cfg)
    storage.save(layer)

    json_path = _user_dir(cfg) / "memory_texts.json"
    json_path.write_text("{ this is : not valid json ]")

    corrupt = MemoryLayer(cfg, FakeEmbedder(cfg))
    # The npz is fine but the JSON sidecar is broken -> guarded load returns False.
    assert storage.load(corrupt) is False
    assert corrupt.memories == []


def test_missing_npz_field_falls_back_to_fresh(tmp_path):
    cfg, layer = _make_layer(tmp_path)
    storage = MemoryStorage(cfg)
    storage.save(layer)

    # Overwrite npz with a valid npz that is missing required fields.
    npz_path = _user_dir(cfg) / "memory.npz"
    np.savez(
        npz_path, keys=np.zeros((2, 16), dtype=np.float32)
    )  # no values/strengths/...

    corrupt = MemoryLayer(cfg, FakeEmbedder(cfg))
    assert storage.load(corrupt) is False
    assert corrupt.memories == []


def test_no_saved_state_returns_false(tmp_path):
    cfg = MemoryConfig(
        embedding_dim=16, embedding_model="test", memory_dir=str(tmp_path)
    )
    storage = MemoryStorage(cfg)
    layer = MemoryLayer(cfg, FakeEmbedder(cfg))
    # Nothing was ever saved.
    assert storage.load(layer) is False


def test_save_is_atomic_no_tmp_left_behind(tmp_path):
    cfg, layer = _make_layer(tmp_path)
    storage = MemoryStorage(cfg)
    storage.save(layer)

    # No leftover .tmp files after a successful atomic save.
    leftovers = list(_user_dir(cfg).glob("*.tmp"))
    assert leftovers == []
    # The real artifacts exist.
    assert (_user_dir(cfg) / "memory.npz").exists()
    assert (_user_dir(cfg) / "memory_texts.json").exists()


def test_load_after_corrupt_then_resave_recovers(tmp_path):
    """After a corrupt-load fallback, a subsequent save/load works again."""
    cfg, layer = _make_layer(tmp_path)
    storage = MemoryStorage(cfg)
    storage.save(layer)

    npz_path = _user_dir(cfg) / "memory.npz"
    npz_path.write_bytes(b"broken")

    corrupt = MemoryLayer(cfg, FakeEmbedder(cfg))
    assert storage.load(corrupt) is False

    # Re-save a good state from the original layer and reload cleanly.
    storage.save(layer)
    recovered = MemoryLayer(cfg, FakeEmbedder(cfg))
    assert storage.load(recovered) is True
    assert len(recovered.memories) == 3


def test_malformed_json_is_actually_invalid():
    """Guard: confirm the corruption payload is genuinely unparseable."""
    import pytest

    with pytest.raises(json.JSONDecodeError):
        json.loads("{ this is : not valid json ]")

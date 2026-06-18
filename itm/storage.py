import json
from pathlib import Path

import numpy as np

from itm.config import MemoryConfig
from itm.core import MemoryEntry, MemoryLayer


class MemoryStorage:
    """Persist memory state to disk as .npz (numerical) + .json (text)."""

    def __init__(self, config: MemoryConfig):
        self.config = config

    def _get_user_dir(self) -> Path:
        path = Path(self.config.memory_dir) / self.config.user_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, layer: MemoryLayer) -> str:
        """Save memory state to disk. Returns the directory path."""
        user_dir = self._get_user_dir()
        n = len(layer.memories)

        if n == 0:
            # Save empty state
            ctx = (
                layer.context_vector
                if layer.context_vector is not None
                else np.zeros(layer.config.embedding_dim, dtype=np.float32)
            )
            np.savez(
                user_dir / "memory.npz",
                keys=np.empty((0, layer.config.embedding_dim), dtype=np.float32),
                values=np.empty((0, layer.config.embedding_dim), dtype=np.float32),
                strengths=np.empty(0, dtype=np.float64),
                biases=np.empty(0, dtype=np.float64),
                timestamps=np.empty(0, dtype=np.int64),
                access_counts=np.empty(0, dtype=np.int64),
                meta=np.array([layer.timestep], dtype=np.int64),
                context_vector=ctx,
            )
            with open(user_dir / "memory_texts.json", "w") as f:
                json.dump([], f)
            return str(user_dir)

        keys = np.stack([m.key for m in layer.memories])
        values = np.stack([m.value for m in layer.memories])
        strengths = np.array([m.strength for m in layer.memories], dtype=np.float64)
        biases = np.array([m.bias for m in layer.memories], dtype=np.float64)
        timestamps = np.array([m.timestamp for m in layer.memories], dtype=np.int64)
        access_counts = np.array(
            [m.access_count for m in layer.memories], dtype=np.int64
        )

        # Equation 8: context vector (RNN hidden state)
        ctx = (
            layer.context_vector
            if layer.context_vector is not None
            else np.zeros(layer.config.embedding_dim, dtype=np.float32)
        )

        np.savez(
            user_dir / "memory.npz",
            keys=keys,
            values=values,
            strengths=strengths,
            biases=biases,
            timestamps=timestamps,
            access_counts=access_counts,
            meta=np.array([layer.timestep], dtype=np.int64),
            context_vector=ctx,
        )

        texts = [
            {
                "input": m.input_text,
                "output": m.output_text,
                "access_history": m.access_history,
                "category": m.category,
            }
            for m in layer.memories
        ]
        with open(user_dir / "memory_texts.json", "w") as f:
            json.dump(texts, f, indent=2)

        # Equation 24: save context_at_creation vectors
        ctx_at_creation = []
        for m in layer.memories:
            if m.context_at_creation is not None:
                ctx_at_creation.append(m.context_at_creation)
            else:
                ctx_at_creation.append(
                    np.zeros(layer.config.embedding_dim, dtype=np.float32)
                )
        if ctx_at_creation:
            np.save(
                user_dir / "context_at_creation.npy",
                np.stack(ctx_at_creation),
            )

        # Equations 22-23: save hierarchy
        if layer.config.hierarchy_enabled:
            hierarchy_data = {
                "l2_patterns": [
                    {
                        "children": p.children,
                    }
                    for p in layer.hierarchy.l2_patterns
                ],
                "l3_identities": [
                    {
                        "children": n.children,
                    }
                    for n in layer.hierarchy.l3_identities
                ],
            }
            with open(user_dir / "hierarchy.json", "w") as f:
                json.dump(hierarchy_data, f, indent=2)

        return str(user_dir)

    def load(self, layer: MemoryLayer) -> bool:
        """Load memory state from disk. Returns True if loaded, False if no saved state."""
        user_dir = self._get_user_dir()
        npz_path = user_dir / "memory.npz"
        json_path = user_dir / "memory_texts.json"

        if not npz_path.exists():
            return False

        data = np.load(npz_path)
        keys = data["keys"]
        values = data["values"]
        strengths = data["strengths"]
        biases = data["biases"]
        timestamps = data["timestamps"]
        access_counts = data["access_counts"]
        meta = data["meta"]

        layer.timestep = int(meta[0])

        # Equation 8: restore context vector
        if "context_vector" in data:
            ctx = data["context_vector"].astype(np.float32)
            if np.linalg.norm(ctx) > 1e-8:
                layer.context_vector = ctx
            else:
                layer.context_vector = None
        else:
            layer.context_vector = None

        # Load text sidecar
        texts = []
        if json_path.exists():
            with open(json_path) as f:
                texts = json.load(f)

        # Equation 24: load context_at_creation vectors
        ctx_path = user_dir / "context_at_creation.npy"
        ctx_at_creation = None
        if ctx_path.exists():
            ctx_at_creation = np.load(ctx_path)

        layer.memories = []
        for i in range(len(keys)):
            input_text = texts[i]["input"] if i < len(texts) else ""
            output_text = texts[i]["output"] if i < len(texts) else ""
            # Phase 2 fields (backward-compatible: default if missing)
            access_history = (
                texts[i].get("access_history", []) if i < len(texts) else []
            )
            category = texts[i].get("category", "") if i < len(texts) else ""
            # Phase 4: context at creation
            ctx_snap = None
            if ctx_at_creation is not None and i < len(ctx_at_creation):
                ctx = ctx_at_creation[i].astype(np.float32)
                if np.linalg.norm(ctx) > 1e-8:
                    ctx_snap = ctx
            entry = MemoryEntry(
                key=keys[i].astype(np.float32),
                value=values[i].astype(np.float32),
                strength=float(strengths[i]),
                bias=float(biases[i]),
                timestamp=int(timestamps[i]),
                access_count=int(access_counts[i]),
                input_text=input_text,
                output_text=output_text,
                access_history=access_history,
                category=category,
                context_at_creation=ctx_snap,
            )
            layer.memories.append(entry)

        # Rebuild the memory graph from loaded memories
        layer.graph.rebuild(layer.memories)

        # Equation 19: Rebuild sparse lexical index
        if layer.config.sparse_recall_enabled:
            layer.sparse_index.rebuild([m.input_text for m in layer.memories])

        # Equations 22-23: Rebuild hierarchy from saved data
        if layer.config.hierarchy_enabled:
            hierarchy_path = user_dir / "hierarchy.json"
            if hierarchy_path.exists():
                with open(hierarchy_path) as f:
                    h_data = json.load(f)
                # Rebuild hierarchy by running consolidate with saved structure
                layer.hierarchy.consolidate(layer.memories)

        return True

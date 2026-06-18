import asyncio

import numpy as np

from itm.config import MemoryConfig


class EmbeddingService:
    """Local embedding backend using sentence-transformers.

    Default model: BAAI/bge-m3 (1024-dim, multi-lingual, multi-granularity).
    Runs entirely on-device: no API calls, no cost, no latency.

    Supports both sync and async:
        vec = embedder.embed("hello")                  # sync
        vec = await embedder.embed_async("hello")      # async (thread pool)

    Requires: pip install sentence-transformers
    """

    def __init__(self, config: MemoryConfig):
        import logging
        import os

        # Silence HF Hub warnings and progress bars during model load
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

        from sentence_transformers import SentenceTransformer

        self.config = config
        self._cache: dict[str, np.ndarray] = {}
        self._model = SentenceTransformer(config.embedding_model)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string, returning a numpy array."""
        if text in self._cache:
            return self._cache[text]

        vec = self._model.encode(text, normalize_embeddings=True)
        vec = np.array(vec, dtype=np.float32)
        self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts, using cache for known strings."""
        uncached_texts = []
        uncached_indices = []

        results = [None] * len(texts)
        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            vecs = self._model.encode(uncached_texts, normalize_embeddings=True)
            for j, vec in enumerate(vecs):
                vec = np.array(vec, dtype=np.float32)
                idx = uncached_indices[j]
                results[idx] = vec
                self._cache[uncached_texts[j]] = vec

        return results

    async def embed_async(self, text: str) -> np.ndarray:
        """Async embed — runs encoding in thread pool (PyTorch releases GIL)."""
        if text in self._cache:
            return self._cache[text]
        return await asyncio.to_thread(self.embed, text)

    async def embed_batch_async(self, texts: list[str]) -> list[np.ndarray]:
        """Async batch embed — runs encoding in thread pool."""
        if all(t in self._cache for t in texts):
            return [self._cache[t] for t in texts]
        return await asyncio.to_thread(self.embed_batch, texts)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def cosine_similarity_matrix(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """Cosine similarity between a query vector and a matrix of key vectors.

        Args:
            query: (d,) vector
            keys: (n, d) matrix

        Returns:
            (n,) array of similarities
        """
        if keys.shape[0] == 0:
            return np.array([], dtype=np.float32)

        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            return np.zeros(keys.shape[0], dtype=np.float32)

        key_norms = np.linalg.norm(keys, axis=1)
        key_norms = np.maximum(key_norms, 1e-8)

        dots = keys @ query
        sims = dots / (key_norms * query_norm)
        return sims.astype(np.float32)

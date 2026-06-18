"""Alternative embedding backends — local (Nomic) and API (OpenAI, Cohere).

Drop-in replacements for EmbeddingService. Same interface:
    embed(text) → np.ndarray
    embed_batch(texts) → list[np.ndarray]
    cosine_similarity / cosine_similarity_matrix (inherited as static)

Usage:
    from itm.embeddings_api import NomicEmbedding, OpenAIEmbedding, CohereEmbedding

    embedder = NomicEmbedding(config)    # local, no API key
    embedder = OpenAIEmbedding(config)   # needs OPENAI_API_KEY
    embedder = CohereEmbedding(config)   # needs COHERE_API_KEY or CO_API_KEY
"""

from __future__ import annotations

import os

import numpy as np

from itm.config import MemoryConfig
from itm.embeddings import EmbeddingService


class NomicEmbedding(EmbeddingService):
    """Nomic nomic-embed-text-v1.5 (768-dim, wider similarity range than BGE-M3).

    Local model via sentence-transformers. Requires trust_remote_code=True
    and text prefixes ("search_document: " / "search_query: ") for best results.
    """

    def __init__(self, config: MemoryConfig):
        from sentence_transformers import SentenceTransformer

        self.config = config
        self._cache: dict[str, np.ndarray] = {}
        self._model = SentenceTransformer(
            config.embedding_model, trust_remote_code=True
        )

    def embed(self, text: str, prefix: str = "search_document: ") -> np.ndarray:
        cache_key = prefix + text
        if cache_key in self._cache:
            return self._cache[cache_key]

        vec = self._model.encode(prefix + text, normalize_embeddings=True)
        vec = np.array(vec, dtype=np.float32)
        self._cache[cache_key] = vec
        return vec

    def embed_batch(
        self, texts: list[str], prefix: str = "search_document: "
    ) -> list[np.ndarray]:
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            cache_key = prefix + text
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            prefixed = [prefix + t for t in uncached_texts]
            vecs = self._model.encode(prefixed, normalize_embeddings=True)
            for j, vec in enumerate(vecs):
                vec = np.array(vec, dtype=np.float32)
                idx = uncached_indices[j]
                results[idx] = vec
                self._cache[prefix + uncached_texts[j]] = vec

        return results


class OpenAIEmbedding(EmbeddingService):
    """OpenAI text-embedding-3-large (3072-dim, widest similarity range)."""

    def __init__(self, config: MemoryConfig):
        # Skip parent __init__ (loads sentence-transformers)
        self.config = config
        self._cache: dict[str, np.ndarray] = {}

        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self._client = OpenAI(api_key=api_key)
        self._model = config.embedding_model

    def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]

        resp = self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        # Normalize to unit length (consistent with BGE-M3)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
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
            resp = self._client.embeddings.create(
                model=self._model,
                input=uncached_texts,
            )
            for j, item in enumerate(resp.data):
                vec = np.array(item.embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    vec = vec / norm
                idx = uncached_indices[j]
                results[idx] = vec
                self._cache[uncached_texts[j]] = vec

        return results


class CohereEmbedding(EmbeddingService):
    """Cohere embed-v4.0 (1024-dim, excellent entity discrimination)."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._cache: dict[str, np.ndarray] = {}

        import cohere

        api_key = os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY or CO_API_KEY environment variable not set"
            )

        self._client = cohere.Client(api_key)
        self._model = config.embedding_model

    def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]

        resp = self._client.embed(
            texts=[text],
            model=self._model,
            input_type="search_document",
            embedding_types=["float"],
        )
        vec = np.array(resp.embeddings.float_[0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
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
            resp = self._client.embed(
                texts=uncached_texts,
                model=self._model,
                input_type="search_document",
                embedding_types=["float"],
            )
            for j, emb in enumerate(resp.embeddings.float_):
                vec = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    vec = vec / norm
                idx = uncached_indices[j]
                results[idx] = vec
                self._cache[uncached_texts[j]] = vec

        return results

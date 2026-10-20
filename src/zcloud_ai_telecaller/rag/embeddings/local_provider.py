"""Local-first ONNX embedding provider using fastembed with zero external cloud dependencies.

Optimized for sub-30ms CPU vector generation and multilingual voice inquiries.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

from zcloud_ai_telecaller.rag.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingTimeoutError,
)

logger = logging.getLogger("zcloud_ai_telecaller.rag.embeddings.local")

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
MULTILINGUAL_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LocalOnnxEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding generator powered by quantized ONNX models on CPU."""

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        cache_dir: Optional[str] = None,
        max_cache_entries: int = 500,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._max_cache_entries = max_cache_entries
        self._embedder = None
        self._dimension: Optional[int] = None
        self._query_cache: dict[str, list[float]] = {}
        self._is_initialized = False

    @property
    def provider_name(self) -> str:
        return "local_onnx"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_initialized()
        return self._dimension or 384

    def _ensure_initialized(self) -> None:
        """Lazy-loads ONNX model into CPU memory."""
        if self._is_initialized and self._embedder is not None:
            return

        try:
            from fastembed import TextEmbedding

            logger.info("Initializing local ONNX embedding model: %s", self._model_name)
            t0 = time.perf_counter()
            self._embedder = TextEmbedding(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
            )
            # Probe dimension
            sample = list(self._embedder.embed(["test"]))
            self._dimension = len(sample[0])
            self._is_initialized = True
            logger.info(
                "Local ONNX embedding model ready in %.1fms (dimension: %d)",
                (time.perf_counter() - t0) * 1000,
                self._dimension,
            )
        except Exception as exc:
            logger.error("Failed to load local ONNX model %s: %s", self._model_name, exc)
            raise EmbeddingProviderError(
                f"Local ONNX embedding model '{self._model_name}' initialization failed: {exc}",
                provider=self.provider_name,
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of documents."""
        if not texts:
            return []

        self._ensure_initialized()
        try:
            assert self._embedder is not None
            embeddings = list(self._embedder.embed(texts))
            return [list(map(float, vec)) for vec in embeddings]
        except Exception as exc:
            logger.error("Document embedding generation failed: %s", exc)
            raise EmbeddingProviderError(
                f"Error embedding documents: {exc}",
                provider=self.provider_name,
            ) from exc

    def embed_query(self, query: str) -> list[float]:
        """Embeds a caller query with query caching to avoid redundant CPU cycles."""
        norm_q = query.strip().lower()
        if not norm_q:
            raise EmbeddingProviderError("Query cannot be empty", provider=self.provider_name)

        # Check in-memory query cache
        q_hash = hashlib.sha256(norm_q.encode("utf-8")).hexdigest()
        if q_hash in self._query_cache:
            return self._query_cache[q_hash]

        self._ensure_initialized()
        try:
            assert self._embedder is not None
            vecs = list(self._embedder.embed([norm_q]))
            float_vec = [float(x) for x in vecs[0]]

            # Bound cache size
            if len(self._query_cache) >= self._max_cache_entries:
                # Evict first 50 entries
                keys_to_evict = list(self._query_cache.keys())[:50]
                for k in keys_to_evict:
                    del self._query_cache[k]

            self._query_cache[q_hash] = float_vec
            return float_vec
        except Exception as exc:
            logger.error("Query embedding generation failed: %s", exc)
            raise EmbeddingProviderError(
                f"Error embedding query: {exc}",
                provider=self.provider_name,
            ) from exc

    def health_check(self) -> bool:
        """Verifies model is loaded and produces valid vectors."""
        try:
            vec = self.embed_query("health check")
            return len(vec) == self.dimension
        except Exception:
            return False

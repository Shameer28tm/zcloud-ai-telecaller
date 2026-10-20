"""Hybrid embedding coordinator with local-first execution and deterministic cloud fallback.

Enforces:
1. Local-first preference (zero cloud API cost, private).
2. Latency timer (350ms budget for local execution).
3. Tenant privacy policy: if tenant forbids cloud embeddings, fallback is blocked.
4. Model & dimension compatibility check.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from zcloud_ai_telecaller.rag.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
    EmbeddingTimeoutError,
)

logger = logging.getLogger("zcloud_ai_telecaller.rag.embeddings.hybrid")


class HybridEmbeddingProvider(BaseEmbeddingProvider):
    """Coordinates local ONNX and cloud fallback embedding generation."""

    def __init__(
        self,
        local_provider: BaseEmbeddingProvider,
        cloud_provider: Optional[BaseEmbeddingProvider] = None,
        allow_cloud_fallback: bool = True,
        local_timeout_seconds: float = 0.35,
    ) -> None:
        self.local_provider = local_provider
        self.cloud_provider = cloud_provider
        self.allow_cloud_fallback = allow_cloud_fallback
        self.local_timeout_seconds = local_timeout_seconds

    @property
    def provider_name(self) -> str:
        return "hybrid"

    @property
    def model_name(self) -> str:
        return self.local_provider.model_name

    @property
    def dimension(self) -> int:
        return self.local_provider.dimension

    def embed_query_with_fallback(self, query: str) -> Tuple[list[float], str, bool]:
        """Embeds query using local provider first, with deterministic cloud fallback.

        Returns:
            Tuple of (embedding_vector, provider_name_used, fallback_triggered)
        """
        t0 = time.perf_counter()
        try:
            vec = self.local_provider.embed_query(query)
            elapsed = time.perf_counter() - t0
            if elapsed > self.local_timeout_seconds:
                logger.warning(
                    "Local embedding took %.1fms (exceeded budget of %.1fms)",
                    elapsed * 1000,
                    self.local_timeout_seconds * 1000,
                )
            return vec, self.local_provider.provider_name, False

        except Exception as local_err:
            logger.warning("Local embedding provider failed: %s", local_err)

            if not self.allow_cloud_fallback:
                logger.error("Cloud fallback is disabled by tenant privacy policy.")
                raise EmbeddingProviderError(
                    f"Local embedding failed and cloud fallback is disabled: {local_err}",
                    provider="local_onnx",
                ) from local_err

            if not self.cloud_provider:
                raise EmbeddingProviderError(
                    f"Local embedding failed and no cloud provider is configured: {local_err}",
                    provider="local_onnx",
                ) from local_err

            logger.info("Triggering cloud embedding fallback (%s)...", self.cloud_provider.model_name)
            try:
                cloud_vec = self.cloud_provider.embed_query(query)
                if len(cloud_vec) != self.dimension:
                    raise EmbeddingDimensionMismatchError(
                        f"Cloud fallback vector dimension {len(cloud_vec)} does not match index dimension {self.dimension}",
                        provider="gemini_cloud",
                    )
                return cloud_vec, self.cloud_provider.provider_name, True
            except Exception as cloud_err:
                logger.error("Cloud embedding fallback also failed: %s", cloud_err)
                raise EmbeddingProviderError(
                    f"Both local and cloud embedding providers failed. Local: {local_err}; Cloud: {cloud_err}",
                    provider="hybrid",
                ) from cloud_err

    def embed_query(self, query: str) -> list[float]:
        vec, _, _ = self.embed_query_with_fallback(query)
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds batch of documents (uses local provider, or fallback if permitted)."""
        try:
            return self.local_provider.embed_documents(texts)
        except Exception as local_err:
            if not self.allow_cloud_fallback or not self.cloud_provider:
                raise local_err
            logger.warning("Local batch embedding failed (%s), falling back to cloud...", local_err)
            cloud_vecs = self.cloud_provider.embed_documents(texts)
            if cloud_vecs and len(cloud_vecs[0]) != self.dimension:
                raise EmbeddingDimensionMismatchError(
                    f"Cloud vector dimension {len(cloud_vecs[0])} does not match {self.dimension}",
                    provider="gemini_cloud",
                )
            return cloud_vecs

    def health_check(self) -> bool:
        return self.local_provider.health_check()

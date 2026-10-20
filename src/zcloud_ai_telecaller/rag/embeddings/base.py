"""Abstract base interface for RAG embedding providers.

Ensures strict provider abstraction, capability checking, error classification,
and dimension compatibility guards.
"""

from __future__ import annotations

import abc
from typing import Any, Optional


class EmbeddingProviderError(Exception):
    """Base error for all embedding provider issues."""
    def __init__(self, message: str, provider: str = "unknown", is_transient: bool = False):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.is_transient = is_transient


class EmbeddingTimeoutError(EmbeddingProviderError):
    """Raised when an embedding provider query exceeds its latency deadline."""
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, is_transient=True)


class EmbeddingDimensionMismatchError(EmbeddingProviderError):
    """Raised when query and indexed document dimensions do not match."""


class BaseEmbeddingProvider(abc.ABC):
    """Interface for embedding models (local ONNX, cloud Gemini, hybrid)."""

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Name of the provider backend (e.g. 'local_onnx', 'gemini_cloud')."""
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Unique model identifier (e.g. 'BAAI/bge-small-en-v1.5', 'text-embedding-004')."""
        pass

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Vector output dimensionality (e.g. 384, 768)."""
        pass

    @property
    def version(self) -> str:
        """Model or index version string."""
        return "v1"

    @abc.abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generates embedding vectors for a batch of document texts.

        Args:
            texts: List of normalized chunk texts.

        Returns:
            List of float vector arrays of length `self.dimension`.
        """
        pass

    @abc.abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generates an embedding vector for a single search query.

        Args:
            query: Caller query string.

        Returns:
            Float vector array of length `self.dimension`.
        """
        pass

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Verifies model readiness and availability.

        Returns:
            True if provider is ready to serve embeddings, False otherwise.
        """
        pass

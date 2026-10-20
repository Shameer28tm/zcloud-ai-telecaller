"""Cloud fallback embedding provider using Google Gemini Embeddings API.

Adheres to tenant policy constraints and provides configurable dimensions.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from zcloud_ai_telecaller.rag.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingTimeoutError,
)

logger = logging.getLogger("zcloud_ai_telecaller.rag.embeddings.cloud")

DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSION = 384


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Generates embeddings using Google GenAI API (Gemini)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_EMBEDDING_MODEL,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        timeout_seconds: float = 4.0,
    ) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._dimension = dimension
        self._timeout_seconds = timeout_seconds
        self._client = None

    @property
    def provider_name(self) -> str:
        return "gemini_cloud"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self._api_key:
            from zcloud_ai_telecaller.config import get_settings
            self._api_key = get_settings().gemini_api_key

        if not self._api_key:
            raise EmbeddingProviderError(
                "GEMINI_API_KEY is not configured for cloud embedding provider.",
                provider=self.provider_name,
            )

        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
            return self._client
        except Exception as exc:
            logger.error("Failed to initialize google.genai Client: %s", exc)
            raise EmbeddingProviderError(
                f"Failed to initialize Gemini client: {exc}",
                provider=self.provider_name,
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of texts using Gemini API."""
        if not texts:
            return []

        client = self._get_client()
        try:
            from google.genai import types

            config = types.EmbedContentConfig(
                output_dimensionality=self._dimension,
            ) if self._dimension else None

            # GenAI embed_content supports batch contents
            response = client.models.embed_content(
                model=self._model_name,
                contents=texts,
                config=config,
            )

            results: list[list[float]] = []
            if hasattr(response, "embeddings") and response.embeddings:
                for emb in response.embeddings:
                    values = getattr(emb, "values", None) or []
                    results.append([float(v) for v in values])
            elif hasattr(response, "embedding") and response.embedding:
                values = getattr(response.embedding, "values", None) or []
                results.append([float(v) for v in values])

            if len(results) != len(texts):
                logger.warning("Gemini returned %d embeddings for %d texts", len(results), len(texts))

            return results
        except Exception as exc:
            err_str = str(exc).lower()
            is_quota = "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str
            logger.error("Gemini document embedding failed: %s", exc)
            raise EmbeddingProviderError(
                f"Gemini embedding call failed: {exc}",
                provider=self.provider_name,
                is_transient=is_quota,
            ) from exc

    def embed_query(self, query: str) -> list[float]:
        """Embeds a search query with task_type retrieval_query."""
        clean_q = query.strip()
        if not clean_q:
            raise EmbeddingProviderError("Query cannot be empty", provider=self.provider_name)

        client = self._get_client()
        try:
            from google.genai import types

            config = types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self._dimension,
            ) if self._dimension else None

            response = client.models.embed_content(
                model=self._model_name,
                contents=[clean_q],
                config=config,
            )

            if hasattr(response, "embeddings") and response.embeddings:
                values = getattr(response.embeddings[0], "values", None) or []
                return [float(v) for v in values]
            elif hasattr(response, "embedding") and response.embedding:
                values = getattr(response.embedding, "values", None) or []
                return [float(v) for v in values]

            raise EmbeddingProviderError("Empty embedding response from Gemini", provider=self.provider_name)
        except Exception as exc:
            logger.error("Gemini query embedding failed: %s", exc)
            raise EmbeddingProviderError(
                f"Gemini query embedding failed: {exc}",
                provider=self.provider_name,
            ) from exc

    def health_check(self) -> bool:
        """Verifies Gemini client connectivity."""
        try:
            vec = self.embed_query("ping")
            return len(vec) == self.dimension
        except Exception:
            return False

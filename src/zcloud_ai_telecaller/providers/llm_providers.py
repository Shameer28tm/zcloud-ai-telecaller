"""LLM Provider wrappers for Z Cloud AI Receptionist.

Implements BaseLLMProvider with metadata for:
- GeminiLLMProvider: Google Gemini 2.5 Flash / Pro
- OllamaLLMProvider: Local Ollama (Qwen, Llama)
- HybridLLMProvider: Ollama with Gemini fallback
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from livekit.agents import llm
from livekit.plugins import google

from zcloud_ai_telecaller.llm_provider import HybridLLM, OllamaLLM
from zcloud_ai_telecaller.providers.base import (
    BaseLLMProvider,
    ProviderMetadata,
    ProviderType,
)

logger = logging.getLogger("zcloud_ai_telecaller.providers.llm")


class GeminiLLMProvider(google.LLM, BaseLLMProvider):
    """Google Gemini LLM provider with pricing metadata and health checks."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        is_byok: bool = False,
    ) -> None:
        eff_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or "dummy-key-for-initialization"
        )
        super().__init__(
            model=model,
            api_key=eff_key,
            temperature=temperature,
        )
        self._is_byok = is_byok

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_name="gemini",
            provider_type=ProviderType.LLM,
            model=self._opts.model if hasattr(self, "_opts") else "gemini-2.5-flash",
            is_local=False,
            is_byok=self._is_byok,
            cost_per_1k_tokens_usd=0.00015,  # Gemini 2.5 Flash input/output blended
            supported_languages=["en", "ta", "thanglish"],
        )


class OllamaLLMProvider(OllamaLLM, BaseLLMProvider):
    """Local Ollama provider with zero-cost metadata."""

    def __init__(
        self,
        *,
        model: str = "qwen3:4b",
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.7,
        timeout: float = 8.0,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
        )

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_name="ollama",
            provider_type=ProviderType.LLM,
            model=self._model,
            is_local=True,
            is_byok=False,
            cost_per_1k_tokens_usd=0.0,
            supported_languages=["en", "ta"],
        )


class HybridLLMProvider(HybridLLM, BaseLLMProvider):
    """Hybrid LLM provider: Ollama primary with Gemini fallback."""

    def __init__(
        self,
        *,
        primary_ollama: OllamaLLM,
        fallback_gemini: google.LLM,
        attempt_timeout: float = 8.0,
    ) -> None:
        super().__init__(
            primary_ollama=primary_ollama,
            fallback_gemini=fallback_gemini,
            attempt_timeout=attempt_timeout,
        )

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_name="hybrid",
            provider_type=ProviderType.LLM,
            model=f"hybrid({self._primary.model}+{self._fallback._opts.model if hasattr(self._fallback, '_opts') else 'gemini'})",
            is_local=False,
            is_byok=False,
            cost_per_1k_tokens_usd=0.00008,
            supported_languages=["en", "ta", "thanglish"],
        )

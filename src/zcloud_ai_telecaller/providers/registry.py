"""Provider Catalog and Registry for Z Cloud AI Receptionist.

Inspired by Dograh's configuration registry, this module acts as the
single source of truth for provider discovery, model selection, BYOK defaults,
and cost metering.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from zcloud_ai_telecaller.providers.base import (
    BaseLLMProvider,
    BaseSTTProvider,
    BaseTTSProvider,
    ProviderMetadata,
    ProviderType,
)
from zcloud_ai_telecaller.providers.llm_providers import (
    GeminiLLMProvider,
    HybridLLMProvider,
    OllamaLLMProvider,
)
from zcloud_ai_telecaller.providers.stt_providers import (
    AdaptiveMultilingualSTT,
    FasterWhisperSTTProvider,
    SarvamSTTProvider,
)
from zcloud_ai_telecaller.providers.tts_providers import (
    AdaptiveBilingualTTSProvider,
    PiperTTSProvider,
    SarvamTTSProvider,
)

logger = logging.getLogger("zcloud_ai_telecaller.providers.registry")

# Static catalog of all available providers
PROVIDER_CATALOG: dict[ProviderType, dict[str, dict[str, Any]]] = {
    ProviderType.STT: {
        "faster-whisper": {
            "name": "Faster-Whisper (Local)",
            "models": ["tiny", "base", "small", "medium", "large-v3"],
            "default_model": "base",
            "is_local": True,
            "cost_per_minute": 0.0,
            "supports_byok": False,
        },
        "sarvam": {
            "name": "Sarvam AI Saaras v4 (Cloud)",
            "models": ["saaras:v4", "saaras:v2"],
            "default_model": "saaras:v4",
            "is_local": False,
            "cost_per_minute": 0.005,
            "supports_byok": True,
        },
    },
    ProviderType.TTS: {
        "piper": {
            "name": "Piper Neural Voice (Local)",
            "models": ["en_US-lessac-medium", "en_GB-alan-medium"],
            "default_model": "en_US-lessac-medium",
            "is_local": True,
            "cost_per_minute": 0.0,
            "supports_byok": False,
        },
        "sarvam": {
            "name": "Sarvam AI Bulbul (Cloud)",
            "models": ["bulbul:v4-flash", "bulbul:v3"],
            "default_model": "bulbul:v4-flash",
            "is_local": False,
            "cost_per_minute": 0.006,
            "supports_byok": True,
        },
        "adaptive-bilingual": {
            "name": "Adaptive Bilingual (Piper + Sarvam)",
            "models": ["adaptive"],
            "default_model": "adaptive",
            "is_local": False,
            "cost_per_minute": 0.003,
            "supports_byok": True,
        },
    },
    ProviderType.LLM: {
        "gemini": {
            "name": "Google Gemini (Cloud)",
            "models": ["gemini-3.8-live", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"],
            "default_model": "gemini-3.8-live",
            "is_local": False,
            "cost_per_1k_tokens": 0.00015,
            "supports_byok": True,
        },
        "ollama": {
            "name": "Ollama Local AI",
            "models": ["qwen3:4b", "llama3.2:3b", "mistral:7b"],
            "default_model": "qwen3:4b",
            "is_local": True,
            "cost_per_1k_tokens": 0.0,
            "supports_byok": False,
        },
        "hybrid": {
            "name": "Hybrid (Ollama + Gemini)",
            "models": ["hybrid"],
            "default_model": "hybrid",
            "is_local": False,
            "cost_per_1k_tokens": 0.00008,
            "supports_byok": True,
        },
    },
}


class ProviderRegistry:
    """Factory and registry for resolving speech and language providers."""

    @staticmethod
    def create_stt(
        provider_name: str = "faster-whisper",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        language: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        is_byok: bool = False,
    ) -> BaseSTTProvider:
        clean = (provider_name or "").lower().strip()
        if clean in ("sarvam", "saaras"):
            return SarvamSTTProvider(
                api_key=api_key,
                model=model or "saaras:v4",
                language_code=language or "ta-IN",
                is_byok=is_byok,
            )
        return FasterWhisperSTTProvider(
            model_size_or_path=model or "base",
            device=device,
            compute_type=compute_type,
            language=language,
        )

    @staticmethod
    def create_tts(
        provider_name: str = "adaptive-bilingual",
        voice_id: Optional[str] = None,
        sarvam_api_key: Optional[str] = None,
        sarvam_model: Optional[str] = None,
        route_thanglish_to_sarvam: bool = False,
        is_byok: bool = False,
    ) -> BaseTTSProvider:
        clean = (provider_name or "").lower().strip()
        eff_sarvam_key = sarvam_api_key or os.environ.get("SARVAM_API_KEY")

        if clean == "sarvam":
            return SarvamTTSProvider(
                api_key=eff_sarvam_key,
                model=sarvam_model or "bulbul:v4-flash",
                is_byok=is_byok,
            )
        elif clean == "piper":
            return PiperTTSProvider(
                voice_name=voice_id or "en_US-lessac-medium",
            )
        else:
            # Adaptive Bilingual default
            piper = PiperTTSProvider(voice_name=voice_id or "en_US-lessac-medium")
            sarvam = (
                SarvamTTSProvider(
                    api_key=eff_sarvam_key,
                    model=sarvam_model or "bulbul:v4-flash",
                    is_byok=is_byok,
                )
                if eff_sarvam_key
                else None
            )
            return AdaptiveBilingualTTSProvider(
                piper_tts=piper,
                sarvam_tts=sarvam,
                route_thanglish_to_sarvam=route_thanglish_to_sarvam,
            )

    @staticmethod
    def create_llm(
        provider_name: str = "gemini",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        is_byok: bool = False,
    ) -> BaseLLMProvider:
        clean = (provider_name or "").lower().strip()
        if clean in ("ollama", "local"):
            return OllamaLLMProvider(
                model=model or "qwen3:4b",
                base_url=base_url or "http://127.0.0.1:11434",
                temperature=temperature,
            )
        elif clean == "hybrid":
            primary = OllamaLLMProvider(
                model=model or "qwen3:4b",
                base_url=base_url or "http://127.0.0.1:11434",
                temperature=temperature,
            )
            fallback = GeminiLLMProvider(
                model="gemini-2.5-flash",
                api_key=api_key,
                temperature=temperature,
                is_byok=is_byok,
            )
            return HybridLLMProvider(primary_ollama=primary, fallback_gemini=fallback)
        else:
            return GeminiLLMProvider(
                model=model or "gemini-2.5-flash",
                api_key=api_key,
                temperature=temperature,
                is_byok=is_byok,
            )

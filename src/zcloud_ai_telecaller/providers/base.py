"""Shared Provider Interface for Z Cloud AI Receptionist.

Defines the extensible, decoupled base provider classes and metadata
for STT, LLM, and TTS providers across cloud, local, and BYOK configurations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from livekit.agents import llm, stt, tts


class ProviderType(str, Enum):
    STT = "stt"
    TTS = "tts"
    LLM = "llm"


@dataclass
class ProviderMetadata:
    provider_name: str
    provider_type: ProviderType
    model: str
    is_local: bool = False
    is_byok: bool = False
    cost_per_minute_usd: float = 0.0
    cost_per_1k_tokens_usd: float = 0.0
    supported_languages: list[str] = None  # e.g. ["en", "ta", "thanglish"]

    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["en"]


class BaseProvider(ABC):
    """Base interface for all speech and language providers."""

    @property
    @abstractmethod
    def metadata(self) -> ProviderMetadata:
        """Returns provider capability and pricing metadata."""
        pass

    async def health_check(self) -> bool:
        """Verifies connectivity or model readiness without incurring billable costs."""
        return True


class BaseSTTProvider(stt.STT, BaseProvider):
    """Abstract base class for Speech-To-Text providers."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.STT


class BaseTTSProvider(tts.TTS, BaseProvider):
    """Abstract base class for Text-To-Speech providers."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.TTS


class BaseLLMProvider(llm.LLM, BaseProvider):
    """Abstract base class for LLM providers."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LLM

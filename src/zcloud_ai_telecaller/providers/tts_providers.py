"""TTS Provider implementations for Z Cloud AI Receptionist.

Includes:
- PiperTTSProvider: Zero-cost local neural TTS using Piper.
- SarvamTTSProvider: Sarvam AI Bulbul v4-flash / v3 for Tamil and Thanglish voice synthesis.
- AdaptiveBilingualTTSProvider: Smart language & script routing between Piper and Sarvam.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from livekit.agents import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    tts,
)

from zcloud_ai_telecaller.local_tts import (
    DEFAULT_VOICE,
    PiperTTS,
    SarvamTTS,
    AdaptiveBilingualTTS as LegacyAdaptiveBilingualTTS,
)
from zcloud_ai_telecaller.providers.base import (
    BaseTTSProvider,
    ProviderMetadata,
    ProviderType,
)

logger = logging.getLogger("zcloud_ai_telecaller.providers.tts")

THANGLISH_MARKERS = {
    "vanakkam", "ungalluku", "ungalukku", "enna", "venum", "vendum", "sollunga",
    "nandri", "romba", "ippo", "naalaikku", "nalaikku", "inniku", "epdi", "eppadi",
    "illai", "aama", "teriyum", "varum", "kudunga", "paakalaam", "irukku", "pannanum",
    "pannidalam", "paravaallai", "solreenga", "ketkuthu", "sari", "vaanga", "ponga",
}


def is_tamil_unicode(text: str) -> bool:
    """Checks whether the text contains native Tamil Unicode characters."""
    return any("\u0b80" <= c <= "\u0bff" for c in text)


def is_thanglish_latin(text: str) -> bool:
    """Checks whether Latin-script text contains recognizable Thanglish conversational tokens."""
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    match_count = sum(1 for t in tokens if t in THANGLISH_MARKERS)
    return match_count >= 1


class PiperTTSProvider(PiperTTS, BaseTTSProvider):
    """Local TTS provider utilizing Piper neural text-to-speech with metadata."""

    def __init__(
        self,
        *,
        voice_name: str = DEFAULT_VOICE,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        sample_rate: int = 22050,
        use_cuda: bool = False,
    ) -> None:
        super().__init__(
            voice_name=voice_name,
            model_path=model_path,
            config_path=config_path,
            sample_rate=sample_rate,
            use_cuda=use_cuda,
        )

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_name="piper",
            provider_type=ProviderType.TTS,
            model=getattr(self, "_voice_name", DEFAULT_VOICE),
            is_local=True,
            is_byok=False,
            cost_per_minute_usd=0.0,
            supported_languages=["en"],
        )


class SarvamTTSProvider(SarvamTTS, BaseTTSProvider):
    """Sarvam AI Bulbul TTS provider for native Tamil & Thanglish voice synthesis."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "bulbul:v4-flash",
        language_code: str = "ta-IN",
        speaker: str = "anant",
        sample_rate: int = 22050,
        is_byok: bool = False,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            language_code=language_code,
            speaker=speaker,
            sample_rate=sample_rate,
        )
        self._is_byok = is_byok

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_name="sarvam",
            provider_type=ProviderType.TTS,
            model=self._model,
            is_local=False,
            is_byok=self._is_byok,
            cost_per_minute_usd=0.006,  # Approx ₹0.50/min
            supported_languages=["ta", "thanglish", "en"],
        )


class AdaptiveBilingualTTSProvider(LegacyAdaptiveBilingualTTS, BaseTTSProvider):
    """Adaptive TTS router directing Tamil and optionally Thanglish to Sarvam AI Bulbul,
    and standard English to Piper neural TTS.
    """

    def __init__(
        self,
        *,
        piper_tts: PiperTTS,
        sarvam_tts: Optional[SarvamTTS] = None,
        route_thanglish_to_sarvam: bool = False,
    ) -> None:
        super().__init__(
            piper_tts=piper_tts,
            sarvam_tts=sarvam_tts,
        )
        self.route_thanglish_to_sarvam = route_thanglish_to_sarvam

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_name="adaptive-bilingual",
            provider_type=ProviderType.TTS,
            model=self.model,
            is_local=False,
            is_byok=False,
            cost_per_minute_usd=0.003,
            supported_languages=["en", "ta", "thanglish"],
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        has_tamil_script = is_tamil_unicode(text)
        has_thanglish_tokens = self.route_thanglish_to_sarvam and is_thanglish_latin(text)

        if has_tamil_script or has_thanglish_tokens:
            if self._sarvam_tts and self._sarvam_tts._api_key:
                logger.info(
                    "Routing %s to Sarvam Bulbul TTS",
                    "Tamil script" if has_tamil_script else "Thanglish text",
                )
                return self._sarvam_tts.synthesize(text, conn_options=conn_options)
            else:
                logger.warning(
                    "Tamil/Thanglish text detected for synthesis, but SARVAM_API_KEY is not configured. "
                    "Falling back to Piper local TTS."
                )

        return self._piper_tts.synthesize(text, conn_options=conn_options)

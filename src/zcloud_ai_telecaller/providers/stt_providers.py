"""STT Provider implementations for Z Cloud AI Receptionist.

Includes:
- FasterWhisperSTTProvider: Zero-cost local STT using faster-whisper.
- SarvamSTTProvider: Sarvam Saaras v4 STT for Tamil, Thanglish, and Indic code-mix.
- AdaptiveMultilingualSTT: Dynamic router switching between local Whisper and cloud Sarvam.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from typing import Any, Optional

import httpx
from livekit import rtc
from livekit.agents import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from zcloud_ai_telecaller.local_stt import FasterWhisperSTT
from zcloud_ai_telecaller.providers.base import (
    BaseSTTProvider,
    ProviderMetadata,
    ProviderType,
)

logger = logging.getLogger("zcloud_ai_telecaller.providers.stt")


class FasterWhisperSTTProvider(FasterWhisperSTT, BaseSTTProvider):
    """Local STT provider implementing BaseSTTProvider with zero-cost metadata."""

    def __init__(
        self,
        *,
        model_size_or_path: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_size_or_path=model_size_or_path,
            device=device,
            compute_type=compute_type,
            language=language,
            initial_prompt=initial_prompt,
        )

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_name="faster-whisper",
            provider_type=ProviderType.STT,
            model=self._model_size_or_path,
            is_local=True,
            is_byok=False,
            cost_per_minute_usd=0.0,
            supported_languages=["en", "ta"],
        )


class SarvamSTTProvider(BaseSTTProvider):
    """Sarvam AI Saaras v4 STT provider for high-accuracy Tamil & Thanglish recognition."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "saaras:v4",
        language_code: str = "ta-IN",
        is_byok: bool = False,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self._model = model
        self._language_code = language_code
        self._is_byok = is_byok

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "sarvam"

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_name="sarvam",
            provider_type=ProviderType.STT,
            model=self._model,
            is_local=False,
            is_byok=self._is_byok,
            cost_per_minute_usd=0.005,  # Approx ₹0.40/min
            supported_languages=["ta", "thanglish", "en"],
        )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        if not self._api_key:
            logger.warning("SarvamSTT called without API key; returning empty transcript.")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[],
            )

        if hasattr(buffer, "to_wav_bytes"):
            wav_bytes = buffer.to_wav_bytes()
        elif isinstance(buffer, (list, tuple)):
            if not buffer:
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[],
                )
            combined_frame = rtc.combine_audio_frames(buffer)
            wav_bytes = combined_frame.to_wav_bytes()
        else:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[],
            )

        # Sarvam Speech-to-Text API multipart upload
        url = "https://api.sarvam.ai/speech-to-text"
        headers = {
            "api-subscription-key": self._api_key,
        }
        data = {
            "model": self._model,
            "language_code": self._language_code,
        }
        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, headers=headers, data=data, files=files)
                resp.raise_for_status()
                result = resp.json()

            transcript = result.get("transcript", "").strip()
            detected_lang = result.get("language_code", self._language_code)

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        language=detected_lang,
                        text=transcript,
                    )
                ],
            )
        except Exception as exc:
            logger.error("Sarvam STT transcription failed: %s", exc)
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[],
            )


class AdaptiveMultilingualSTT(BaseSTTProvider):
    """Adaptive STT router: Uses Sarvam Saaras when available and preferred,
    with automatic local Faster-Whisper fallback to guarantee zero service interruption.
    """

    def __init__(
        self,
        *,
        primary_stt: BaseSTTProvider,
        fallback_stt: Optional[BaseSTTProvider] = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._primary = primary_stt
        self._fallback = fallback_stt or FasterWhisperSTTProvider()

    @property
    def model(self) -> str:
        return f"adaptive({self._primary.model}+{self._fallback.model})"

    @property
    def provider(self) -> str:
        return "adaptive-multilingual"

    @property
    def metadata(self) -> ProviderMetadata:
        return self._primary.metadata

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        try:
            res = await self._primary._recognize_impl(
                buffer, language=language, conn_options=conn_options
            )
            if res and res.alternatives and res.alternatives[0].text:
                return res
        except Exception as exc:
            logger.warning("Primary STT failed (%s); failing over to fallback STT", exc)

        return await self._fallback._recognize_impl(
            buffer, language=language, conn_options=conn_options
        )

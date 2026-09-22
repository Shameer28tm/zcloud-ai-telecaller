"""Local Speech-To-Text (STT) implementation using faster-whisper.

Provides zero-cost local speech recognition compatible with LiveKit Agents.
"""

import asyncio
import io
import logging
from typing import Optional

from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

logger = logging.getLogger("zcloud_ai_telecaller.stt")


class FasterWhisperSTT(stt.STT):
    """Local STT provider using faster-whisper (CTranslate2).

    Runs locally at zero API cost. Designed to be wrapped with
    `livekit.agents.stt.StreamAdapter` to provide streaming STT capabilities
    to LiveKit voice sessions.
    """

    def __init__(
        self,
        *,
        model_size_or_path: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        language: Optional[str] = "en",
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._model_size_or_path = model_size_or_path
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model: Optional[WhisperModel] = None
        self._lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return f"faster-whisper-{self._model_size_or_path}"

    @property
    def provider(self) -> str:
        return "faster-whisper"

    def _load_model(self) -> WhisperModel:
        """Loads or returns the cached WhisperModel instance."""
        if self._model is None:
            logger.info(
                "Loading faster-whisper model: %s (device=%s, compute_type=%s)",
                self._model_size_or_path,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size_or_path,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info("faster-whisper model loaded successfully: %s", self._model_size_or_path)
        return self._model

    def prewarm(self) -> None:
        """Preload the model weights during worker initialization."""
        self._load_model()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Transcribes the input audio buffer using local faster-whisper."""
        # Convert audio buffer to WAV format
        if isinstance(buffer, rtc.AudioFrame):
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

        resolved_lang = language if language is not NOT_GIVEN and language else self._language

        def _sync_transcribe() -> tuple[str, str]:
            model = self._load_model()
            audio_stream = io.BytesIO(wav_bytes)
            segments, info = model.transcribe(
                audio_stream,
                language=resolved_lang,
                beam_size=1,
                condition_on_previous_text=False,
            )
            transcription = " ".join(seg.text.strip() for seg in segments).strip()
            detected_language = getattr(info, "language", resolved_lang) or "en"
            return transcription, detected_language

        text, lang = await asyncio.to_thread(_sync_transcribe)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=lang,
                    text=text,
                )
            ],
        )

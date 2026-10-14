"""Local Text-To-Speech (TTS) implementation using Piper neural voice synthesis.

Provides fast, high-quality, local speech synthesis at ₹0 API cost.
"""

import asyncio
import io
import logging
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from livekit.agents import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    tts,
)
from piper import PiperVoice

logger = logging.getLogger("zcloud_ai_telecaller.tts")

DEFAULT_VOICE = "en_US-lessac-medium"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/piper")


def resolve_voice_urls(voice_name: str) -> tuple[str, str]:
    """Resolves official Hugging Face URLs for the specified Piper voice model.

    Format convention: {lang}_{country}-{speaker}-{quality}
    Example: en_US-lessac-medium -> en/en_US/lessac/medium/en_US-lessac-medium
    """
    parts = voice_name.split("-")
    if len(parts) >= 3:
        locale = parts[0]
        lang = locale.split("_")[0]
        speaker = parts[1]
        quality = parts[2]
        base_url = (
            f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            f"{lang}/{locale}/{speaker}/{quality}/{voice_name}"
        )
    else:
        base_url = (
            f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            f"en/en_US/lessac/medium/{voice_name}"
        )
    return f"{base_url}.onnx", f"{base_url}.onnx.json"


def ensure_piper_model(
    voice_name: str = DEFAULT_VOICE,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> tuple[str, str]:
    """Ensures that the Piper ONNX model and JSON config exist locally.

    Downloads the model files from Hugging Face if not already present.
    """
    target_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = Path(model_path) if model_path else target_dir / f"{voice_name}.onnx"
    resolved_config = Path(config_path) if config_path else target_dir / f"{voice_name}.onnx.json"

    if not resolved_model.exists() or not resolved_config.exists():
        onnx_url, json_url = resolve_voice_urls(voice_name)
        if not resolved_config.exists():
            logger.info("Downloading Piper voice config for %s...", voice_name)
            urllib.request.urlretrieve(json_url, str(resolved_config))
        if not resolved_model.exists():
            logger.info("Downloading Piper voice model for %s...", voice_name)
            urllib.request.urlretrieve(onnx_url, str(resolved_model))
        logger.info("Piper voice model ready: %s", resolved_model)

    return str(resolved_model), str(resolved_config)


class PiperTTS(tts.TTS):
    """Local TTS provider utilizing Piper neural text-to-speech.

    Compatible with the LiveKit Agents TTS interface.
    """

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
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._voice_name = voice_name
        self._model_path = model_path
        self._config_path = config_path
        self._use_cuda = use_cuda
        self._voice: Optional[PiperVoice] = None

    @property
    def model(self) -> str:
        return f"piper-{self._voice_name}"

    @property
    def provider(self) -> str:
        return "piper"

    def _load_voice(self) -> PiperVoice:
        """Loads or retrieves the initialized PiperVoice instance."""
        if self._voice is None:
            model_file, config_file = ensure_piper_model(
                voice_name=self._voice_name,
                model_path=self._model_path,
                config_path=self._config_path,
            )
            logger.info("Loading Piper voice: %s", model_file)
            self._voice = PiperVoice.load(
                model_path=model_file,
                config_path=config_file,
                use_cuda=self._use_cuda,
            )
            if hasattr(self._voice.config, "sample_rate"):
                self._sample_rate = self._voice.config.sample_rate
            logger.info("Piper voice initialized: %s (sample_rate=%d)", self._voice_name, self._sample_rate)
        return self._voice

    def prewarm(self) -> None:
        """Pre-warms and caches the Piper neural voice model in memory."""
        self._load_voice()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return PiperChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class PiperChunkedStream(tts.ChunkedStream):
    """Chunked synthesis stream implementation for Piper local TTS."""

    def __init__(
        self,
        *,
        tts: PiperTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._piper_tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = str(uuid.uuid4())
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._piper_tts.sample_rate,
            num_channels=self._piper_tts.num_channels,
            mime_type="audio/raw",
        )

        def _synthesize_chunks() -> list[bytes]:
            voice = self._piper_tts._load_voice()
            chunks = voice.synthesize(self._input_text)
            return [chunk.audio_int16_bytes for chunk in chunks if chunk.audio_int16_bytes]

        pcm_chunks = await asyncio.to_thread(_synthesize_chunks)
        for chunk in pcm_chunks:
            output_emitter.push(chunk)
        output_emitter.flush()


class SarvamTTS(tts.TTS):
    """Cloud TTS provider utilizing Sarvam AI Bulbul neural voices for Indic languages (including Tamil)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "bulbul:v3",
        language_code: str = "ta-IN",
        speaker: str = "shreya",
        sample_rate: int = 22050,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self._model = model
        self._language_code = language_code
        self._speaker = speaker

    @property
    def model(self) -> str:
        return f"sarvam-{self._model}"

    @property
    def provider(self) -> str:
        return "sarvam"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return SarvamChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class SarvamChunkedStream(tts.ChunkedStream):
    """Chunked synthesis stream for Sarvam AI Bulbul TTS."""

    def __init__(
        self,
        *,
        tts: SarvamTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._sarvam_tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        import base64
        import wave
        import httpx

        request_id = str(uuid.uuid4())
        api_key = self._sarvam_tts._api_key
        if not api_key:
            raise ValueError("SARVAM_API_KEY is required for SarvamTTS synthesis")

        url = "https://api.sarvam.ai/text-to-speech"
        headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": self._input_text,
            "speaker": self._sarvam_tts._speaker,
            "model": self._sarvam_tts._model,
            "language_code": self._sarvam_tts._language_code,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        audios = data.get("audios", [])
        if not audios:
            output_emitter.flush()
            return

        wav_bytes = base64.b64decode(audios[0])
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            pcm_data = wf.readframes(wf.getnframes())

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=sample_rate,
            num_channels=num_channels,
            mime_type="audio/raw",
        )
        output_emitter.push(pcm_data)
        output_emitter.flush()


THANGLISH_MARKERS = {
    "vanakkam", "ungalluku", "ungalukku", "enna", "venum", "vendum", "sollunga",
    "nandri", "romba", "ippo", "naalaikku", "nalaikku", "inniku", "epdi", "eppadi",
    "illai", "aama", "teriyum", "varum", "kudunga", "paakalaam", "irukku", "pannanum",
    "pannidalam", "paravaallai", "solreenga", "ketkuthu", "sari", "vaanga", "ponga",
}


def is_thanglish_latin(text: str) -> bool:
    """Checks whether Latin-script text contains recognizable Thanglish conversational tokens."""
    import re
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return any(t in THANGLISH_MARKERS for t in tokens)


class AdaptiveBilingualTTS(PiperTTS):
    """Adaptive TTS router that directs native Tamil Unicode text to Sarvam AI Bulbul (if configured)
    and English/Thanglish text to local Piper neural TTS. Optionally routes Thanglish to Sarvam.
    """

    def __init__(
        self,
        *,
        piper_tts: PiperTTS,
        sarvam_tts: Optional[SarvamTTS] = None,
        route_thanglish_to_sarvam: bool = False,
    ) -> None:
        super().__init__(
            voice_name=getattr(piper_tts, "_voice_name", DEFAULT_VOICE),
            model_path=getattr(piper_tts, "_model_path", None),
            config_path=getattr(piper_tts, "_config_path", None),
            sample_rate=getattr(piper_tts, "sample_rate", 22050),
            use_cuda=getattr(piper_tts, "_use_cuda", False),
        )
        self._piper_tts = piper_tts
        self._sarvam_tts = sarvam_tts
        self.route_thanglish_to_sarvam = route_thanglish_to_sarvam

    @property
    def model(self) -> str:
        if self._sarvam_tts and self._sarvam_tts._api_key:
            return f"adaptive({self._piper_tts.model}+{self._sarvam_tts.model})"
        return self._piper_tts.model

    @property
    def provider(self) -> str:
        return "adaptive-bilingual"

    def prewarm(self) -> None:
        self._piper_tts.prewarm()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        is_tamil = any("\u0b80" <= c <= "\u0bff" for c in text)
        is_thanglish = self.route_thanglish_to_sarvam and is_thanglish_latin(text)
        if is_tamil or is_thanglish:
            if self._sarvam_tts and self._sarvam_tts._api_key:
                logger.info(
                    "Routing %s to Sarvam Bulbul TTS",
                    "Tamil script" if is_tamil else "Thanglish text",
                )
                return self._sarvam_tts.synthesize(text, conn_options=conn_options)
            else:
                logger.warning(
                    "Tamil/Thanglish text detected for speech synthesis, but SARVAM_API_KEY is not configured. "
                    "Piper local TTS will attempt synthesis, but native Tamil pronunciation requires Sarvam Bulbul."
                )
        return self._piper_tts.synthesize(text, conn_options=conn_options)


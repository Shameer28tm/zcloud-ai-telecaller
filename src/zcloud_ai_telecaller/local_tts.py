"""Local Text-To-Speech (TTS) implementation using Piper neural voice synthesis.

Provides fast, high-quality, local speech synthesis at ₹0 API cost.
"""

import asyncio
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

        def _synthesize() -> bytes:
            voice = self._piper_tts._load_voice()
            chunks = voice.synthesize(self._input_text)
            pcm_data = bytearray()
            for chunk in chunks:
                pcm_data.extend(chunk.audio_int16_bytes)
            return bytes(pcm_data)

        raw_pcm = await asyncio.to_thread(_synthesize)
        if raw_pcm:
            output_emitter.push(raw_pcm)
        output_emitter.flush()

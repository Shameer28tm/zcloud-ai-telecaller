"""Tests for the Shared Provider Interface and Provider Registry.

Verifies:
- Provider creation, metadata, and cost attributes
- Sarvam STT & TTS mocking (NO paid API calls)
- Adaptive bilingual and Thanglish routing
- Graceful fallbacks and BYOK flags
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from zcloud_ai_telecaller.providers import (
    AdaptiveBilingualTTSProvider,
    AdaptiveMultilingualSTT,
    FasterWhisperSTTProvider,
    GeminiLLMProvider,
    HybridLLMProvider,
    OllamaLLMProvider,
    PiperTTSProvider,
    ProviderRegistry,
    ProviderType,
    SarvamSTTProvider,
    SarvamTTSProvider,
    is_tamil_unicode,
    is_thanglish_latin,
)


def test_provider_registry_stt_creation():
    """Verify registry creates correct STT providers with metadata."""
    local_stt = ProviderRegistry.create_stt("faster-whisper", model="base")
    assert isinstance(local_stt, FasterWhisperSTTProvider)
    assert local_stt.metadata.is_local is True
    assert local_stt.metadata.cost_per_minute_usd == 0.0

    sarvam_stt = ProviderRegistry.create_stt(
        "sarvam",
        model="saaras:v4",
        api_key="mock-key",
        is_byok=True,
    )
    assert isinstance(sarvam_stt, SarvamSTTProvider)
    assert sarvam_stt.metadata.is_local is False
    assert sarvam_stt.metadata.is_byok is True
    assert sarvam_stt.metadata.cost_per_minute_usd > 0


def test_provider_registry_tts_creation():
    """Verify registry creates correct TTS providers with metadata."""
    piper = ProviderRegistry.create_tts("piper", voice_id="en_US-lessac-medium")
    assert isinstance(piper, PiperTTSProvider)
    assert piper.metadata.is_local is True

    sarvam = ProviderRegistry.create_tts(
        "sarvam",
        sarvam_api_key="mock-key",
        sarvam_model="bulbul:v4-flash",
        is_byok=True,
    )
    assert isinstance(sarvam, SarvamTTSProvider)
    assert sarvam.metadata.is_byok is True

    adaptive = ProviderRegistry.create_tts(
        "adaptive-bilingual",
        sarvam_api_key="mock-key",
        route_thanglish_to_sarvam=True,
    )
    assert isinstance(adaptive, AdaptiveBilingualTTSProvider)
    assert adaptive.route_thanglish_to_sarvam is True


def test_provider_registry_llm_creation():
    """Verify registry creates correct LLM providers with metadata."""
    gemini = ProviderRegistry.create_llm(
        "gemini",
        model="gemini-2.5-flash",
        api_key="mock-gemini-key",
        is_byok=True,
    )
    assert isinstance(gemini, GeminiLLMProvider)
    assert gemini.metadata.is_byok is True
    assert gemini.metadata.provider_type == ProviderType.LLM

    ollama = ProviderRegistry.create_llm("ollama", model="qwen3:4b")
    assert isinstance(ollama, OllamaLLMProvider)
    assert ollama.metadata.is_local is True

    hybrid = ProviderRegistry.create_llm("hybrid", model="qwen3:4b", api_key="mock-key")
    assert isinstance(hybrid, HybridLLMProvider)


def test_thanglish_latin_token_detection():
    """Verify Thanglish detection identifies conversational Tamil in Latin script."""
    assert is_thanglish_latin("Vanakkam, enakku appointment venum") is True
    assert is_thanglish_latin("Nandri, naalaikku paakalaam") is True
    assert is_thanglish_latin("Doctor eppadi irukku?") is True

    # Pure English
    assert is_thanglish_latin("Hello, I would like to book a dental checkup.") is False
    assert is_thanglish_latin("Good morning, what time are you open?") is False


def test_adaptive_tts_thanglish_routing_behavior():
    """Verify that AdaptiveBilingualTTSProvider routes Thanglish to Sarvam when enabled."""
    piper_mock = MagicMock(spec=PiperTTSProvider)
    piper_mock.sample_rate = 22050
    piper_mock.num_channels = 1
    piper_mock.model = "piper-mock"

    sarvam_mock = MagicMock(spec=SarvamTTSProvider)
    sarvam_mock._api_key = "mock-sarvam-key"
    sarvam_mock.sample_rate = 22050
    sarvam_mock.num_channels = 1
    sarvam_mock.model = "sarvam-bulbul"

    # 1. With route_thanglish_to_sarvam = True
    adaptive = AdaptiveBilingualTTSProvider(
        piper_tts=piper_mock,
        sarvam_tts=sarvam_mock,
        route_thanglish_to_sarvam=True,
    )

    adaptive.synthesize("Vanakkam! Ungalukku appointment confirm aagiruchu.")
    sarvam_mock.synthesize.assert_called_once()
    piper_mock.synthesize.assert_not_called()

    sarvam_mock.reset_mock()
    piper_mock.reset_mock()

    # 2. Pure English still goes to Piper
    adaptive.synthesize("Hello! Your appointment has been booked.")
    piper_mock.synthesize.assert_called_once()
    sarvam_mock.synthesize.assert_not_called()


@pytest.mark.asyncio
async def test_sarvam_stt_transcription_mocked():
    """Verify SarvamSTTProvider transcribes audio using mocked HTTP response without paid call."""
    provider = SarvamSTTProvider(
        api_key="mock-sarvam-key",
        model="saaras:v4",
        language_code="ta-IN",
    )

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "transcript": "வணக்கம் எனக்கு பல் பரிசோதனை வேண்டும்",
        "language_code": "ta-IN",
    }
    fake_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = fake_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    # Dummy AudioFrame
    frame_mock = MagicMock()
    frame_mock.to_wav_bytes.return_value = b"RIFFmockwavbytes"

    with patch("httpx.AsyncClient", return_value=mock_client):
        event = await provider._recognize_impl(frame_mock)

    assert event.type.name == "FINAL_TRANSCRIPT"
    assert len(event.alternatives) == 1
    assert event.alternatives[0].text == "வணக்கம் எனக்கு பல் பரிசோதனை வேண்டும்"
    assert event.alternatives[0].language == "ta-IN"

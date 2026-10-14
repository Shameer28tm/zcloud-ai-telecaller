"""Tests for Tamil/Thanglish language understanding, state tracking, and TTS routing."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from livekit.agents import tts
from zcloud_ai_telecaller.conversation_state import ConversationStateManager
from zcloud_ai_telecaller.prompt_builder import build_clinic_system_prompt
from zcloud_ai_telecaller.runtime_client import AgentRuntimeConfig
from zcloud_ai_telecaller.local_stt import FasterWhisperSTT
from zcloud_ai_telecaller.local_tts import AdaptiveBilingualTTS, PiperTTS, SarvamTTS


def test_conversation_state_language_detection():
    """Verify that ConversationStateManager accurately detects and preserves language preferences."""
    mgr = ConversationStateManager(tenant_id="t1", agent_id="a1")
    assert mgr.preferred_language == "en"

    # 1. Native Tamil speech/request
    lang = mgr.update_language_preference_from_input("வணக்கம், எனக்கு ஒரு appointment book பண்ண முடியுமா?")
    assert lang == "ta"
    assert mgr.preferred_language == "ta"

    # 2. Subsequent turn in English or short service name retains 'ta'
    mgr.add_message("user", "Dental cleaning")
    assert mgr.preferred_language == "ta"
    summary = mgr.get_structured_summary()
    assert summary["preferredLanguage"] == "ta"

    # 3. Explicit switch back to English
    lang = mgr.update_language_preference_from_input("Please speak in English from now on.")
    assert lang == "en"
    assert mgr.preferred_language == "en"

    # 4. Thanglish input
    lang = mgr.update_language_preference_from_input("Vanakkam, enakku appointment pannanum.")
    assert lang == "thanglish"
    assert mgr.preferred_language == "thanglish"

    # 5. Dynamic switch from Thanglish to Tamil
    lang = mgr.update_language_preference_from_input("நாளைக்கு டாக்டர் இருப்பாங்களா?")
    assert lang == "ta"
    assert mgr.preferred_language == "ta"

    # 6. Dynamic switch from Tamil back to Thanglish
    lang = mgr.update_language_preference_from_input("Doctor available timings enna nu sollunga?")
    assert lang == "thanglish"
    assert mgr.preferred_language == "thanglish"


def test_prompt_builder_contains_multilingual_rules():
    """Verify that system prompt contains clear rules for Tamil, Thanglish, and language persistence."""
    config = AgentRuntimeConfig(
        agent_id="test-agent",
        agent_version_id="v1",
        tenant_id="test-tenant",
        name="Apollo Dental Clinic",
        prompt="",
        greeting="Vanakkam!",
        clinic_profile={
            "name": "Apollo Dental Clinic",
            "phone": "+91 98765 43210",
            "services": ["Routine Consultation", "Dental Cleaning & Checkup"],
        },
    )
    prompt = build_clinic_system_prompt(config)

    # Must NOT forbid Tamil script
    assert "NEVER output responses in Tamil Unicode script" not in prompt

    # Must contain explicit instructions for Tamil, Thanglish, and language retention
    assert "MULTILINGUAL & CODE-SWITCHING LANGUAGE BEHAVIOR" in prompt
    assert "TAMIL INPUT" in prompt
    assert "TANGLISH INPUT" in prompt
    assert "PRESERVE LANGUAGE PREFERENCE" in prompt


def test_faster_whisper_stt_language_defaults():
    """Verify FasterWhisperSTT defaults to language=None for bilingual auto-detection."""
    stt_instance = FasterWhisperSTT(
        model_size_or_path="base",
        device="cpu",
        compute_type="int8",
        language=None,
        initial_prompt="Vanakkam, Tamil, தமிழ், Thanglish clinic appointment booking",
    )
    assert stt_instance._language is None
    assert stt_instance._initial_prompt is not None
    assert "தமிழ்" in stt_instance._initial_prompt


def test_adaptive_bilingual_tts_routing():
    """Verify AdaptiveBilingualTTS routes Tamil script to Sarvam and English/Thanglish to Piper."""
    piper_mock = MagicMock(spec=PiperTTS)
    piper_mock.sample_rate = 22050
    piper_mock.num_channels = 1
    piper_mock.model = "piper-en_US-lessac-medium"

    sarvam_mock = MagicMock(spec=SarvamTTS)
    sarvam_mock._api_key = "test-sarvam-key"
    sarvam_mock.sample_rate = 22050
    sarvam_mock.num_channels = 1
    sarvam_mock.model = "sarvam-bulbul:v3"

    adaptive = AdaptiveBilingualTTS(piper_tts=piper_mock, sarvam_tts=sarvam_mock)

    # 1. English text -> Piper
    adaptive.synthesize("Hello, how can I help you today?")
    piper_mock.synthesize.assert_called_once()
    sarvam_mock.synthesize.assert_not_called()

    piper_mock.reset_mock()
    sarvam_mock.reset_mock()

    # 2. Thanglish in Latin letters -> Piper
    adaptive.synthesize("Vanakkam! Sure, dental cleaning appointment book pannidalam.")
    piper_mock.synthesize.assert_called_once()
    sarvam_mock.synthesize.assert_not_called()

    piper_mock.reset_mock()
    sarvam_mock.reset_mock()

    # 3. Native Tamil Unicode -> Sarvam
    adaptive.synthesize("வணக்கம்! அப்பல்லோ கிளினிக், உங்களுக்கு என்ன உதவி வேண்டும்?")
    sarvam_mock.synthesize.assert_called_once()
    piper_mock.synthesize.assert_not_called()


def test_adaptive_bilingual_tts_fallback_when_no_sarvam_key():
    """Verify AdaptiveBilingualTTS gracefully falls back to Piper when no Sarvam key is provided."""
    piper_mock = MagicMock(spec=PiperTTS)
    piper_mock.sample_rate = 22050
    piper_mock.num_channels = 1
    piper_mock.model = "piper-en_US-lessac-medium"

    # Sarvam with no API key
    sarvam_mock = MagicMock(spec=SarvamTTS)
    sarvam_mock._api_key = None

    adaptive = AdaptiveBilingualTTS(piper_tts=piper_mock, sarvam_tts=sarvam_mock)

    # Tamil text with no key -> Piper fallback
    adaptive.synthesize("வணக்கம்! அப்பல்லோ கிளினிக்.")
    piper_mock.synthesize.assert_called_once()

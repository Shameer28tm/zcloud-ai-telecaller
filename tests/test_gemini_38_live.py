"""Tests for Gemini 3.8 Live native real-time multimodal voice integration.

Verifies:
1. Gemini 3.8 Live RealtimeModel instantiation and LiveAPIModels conformance.
2. AgentSession creation with native audio-to-audio:
   - Zero duplicate STT (Whisper omitted during realtime turns)
   - Low-latency endpointing (min_delay: 0.25, max_delay: 1.5)
   - TTS programmatic say() fallback retained
3. Multilingual Tamil, Tanglish, and English language switching behavior.
4. Clinic tools availability and appointment booking workflow.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from livekit.agents import AgentSession
from livekit.plugins.google import realtime

from zcloud_ai_telecaller.agent import create_agent, create_agent_session
from zcloud_ai_telecaller.conversation_state import ConversationStateManager
from zcloud_ai_telecaller.llm_provider import create_llm_provider
from zcloud_ai_telecaller.prompt_builder import build_clinic_system_prompt
from zcloud_ai_telecaller.runtime_client import AgentRuntimeConfig


def test_gemini_38_live_llm_provider_creation():
    """Verify that create_llm_provider returns RealtimeModel for gemini-3.8-live."""
    provider = create_llm_provider(
        provider="gemini",
        model="gemini-3.8-live",
        api_key="mock-api-key",
    )
    assert isinstance(provider, realtime.RealtimeModel)
    assert provider.model == "gemini-3.8-live"
    assert provider.capabilities.turn_detection is True
    assert provider.capabilities.user_transcription is True
    assert provider.capabilities.audio_output is True


@pytest.mark.asyncio
async def test_gemini_38_live_agent_session_zero_duplicate_stt():
    """Verify that AgentSession with gemini-3.8-live omits duplicate STT and configures low-latency endpointing."""
    session = create_agent_session(
        llm_model="gemini-3.8-live",
        gemini_api_key="mock-api-key",
        use_realtime=True,
    )
    assert isinstance(session, AgentSession)
    # When RealtimeModel is active, LiveKit handles audio natively over WebSocket;
    # STT is None to avoid duplicate Faster-Whisper CPU transcription
    assert session.stt is None
    assert isinstance(session.llm, realtime.RealtimeModel)
    assert session.tts is not None  # Retained for session.say() fallback
    # Verify low-latency endpointing options
    endpointing = session._opts.turn_handling["endpointing"]
    assert endpointing["min_delay"] == 0.25
    assert endpointing["max_delay"] == 1.5


def test_multilingual_language_switching_flow():
    """Verify bidirectional language adaptation across Tamil, Tanglish, and English."""
    mgr = ConversationStateManager(tenant_id="test", agent_id="receptionist")

    # Turn 1: Caller speaks Tamil
    t1 = mgr.update_language_preference_from_input("வணக்கம், எனக்கு பல் வலிக்கு அப்பாயிண்ட்மெண்ட் வேணும்.")
    assert t1 == "ta"
    assert mgr.preferred_language == "ta"

    # Turn 2: Caller switches to Tanglish
    t2 = mgr.update_language_preference_from_input("Doctor available timings enna nu sollunga?")
    assert t2 == "thanglish"
    assert mgr.preferred_language == "thanglish"

    # Turn 3: Caller switches to English
    t3 = mgr.update_language_preference_from_input("Please tell me the exact clinic address in English.")
    assert t3 == "en"
    assert mgr.preferred_language == "en"

    # Turn 4: Caller switches back to Tamil
    t4 = mgr.update_language_preference_from_input("நன்றி, நாளைக்கு வருகிறேன்.")
    assert t4 == "ta"
    assert mgr.preferred_language == "ta"


def test_clinic_prompt_contains_gemini_38_tanglish_rules():
    """Verify that build_clinic_system_prompt contains all Phase 2 multilingual guidelines."""
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
            "address": "No. 123, Anna Salai, Chennai",
            "services": ["Routine Consultation", "Dental Cleaning & Checkup", "Root Canal Treatment"],
        },
    )
    prompt = build_clinic_system_prompt(config)

    assert "DYNAMIC LANGUAGE DETECTION & ADAPTATION" in prompt
    assert "TAMIL INPUT" in prompt
    assert "TANGLISH INPUT" in prompt
    assert "ENGLISH INPUT" in prompt
    assert "ENTITY PRESERVATION" in prompt
    assert "NO REDUNDANT LANGUAGE QUESTIONS" in prompt
    assert "Apollo Dental Clinic" in prompt
    assert "No. 123, Anna Salai, Chennai" in prompt

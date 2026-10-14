"""Regression tests for Gemini Realtime model selection, error 1008 recovery,
fallback pipeline, greeting playback, turn isolation, and session continuity.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from livekit.agents import AgentSession, stt
from livekit.plugins import google
from livekit.plugins.google import realtime

from zcloud_ai_telecaller.agent import (
    AGENT_INSTRUCTIONS,
    GREETING_MESSAGE,
    create_agent,
    create_agent_session,
)
from zcloud_ai_telecaller.conversation_state import ConversationStateManager
from zcloud_ai_telecaller.llm_provider import (
    DEFAULT_GEMINI_REALTIME_MODEL,
    DEFAULT_GEMINI_TEXT_MODEL,
    VERIFIED_GEMINI_REALTIME_MODELS,
    create_llm_provider,
)
from zcloud_ai_telecaller.runtime_client import (
    AgentRuntimeConfig,
    CallSessionAccumulator,
)


# ==============================================================================
# Group 1: Model Selection Verification
# ==============================================================================

def test_model_selection_realtime_with_text_model_input():
    """Verify that requesting realtime with 'gemini-2.5-flash' safely maps to verified Gemini Live model."""
    provider = create_llm_provider(
        provider="gemini",
        model="gemini-2.5-flash",
        api_key="mock-key",
        use_realtime=True,
    )
    assert isinstance(provider, realtime.RealtimeModel)
    assert provider.model in VERIFIED_GEMINI_REALTIME_MODELS
    assert provider.model == DEFAULT_GEMINI_REALTIME_MODEL  # "gemini-3.8-live"


def test_model_selection_cascaded_text_model():
    """Verify that requesting standard cascaded pipeline with 'gemini-2.5-flash' returns google.LLM."""
    provider = create_llm_provider(
        provider="gemini",
        model="gemini-2.5-flash",
        api_key="mock-key",
        use_realtime=False,
    )
    assert isinstance(provider, google.LLM)
    assert provider.model == "gemini-2.5-flash"


def test_model_selection_live_model_with_use_realtime_false():
    """Verify that passing 'gemini-3.8-live' with use_realtime=False safely maps to text LLM to prevent 400 error."""
    provider = create_llm_provider(
        provider="gemini",
        model="gemini-3.8-live",
        api_key="mock-key",
        use_realtime=False,
    )
    assert isinstance(provider, google.LLM)
    assert provider.model == "gemini-2.5-flash"


def test_verified_realtime_models_coverage():
    """Verify that all officially verified Gemini Live models instantiate RealtimeModel."""
    for model_name in [
        "gemini-3.8-live",
        "gemini-2.5-flash-native-audio-preview-12-2025",
        "gemini-3.1-flash-live-preview",
        "gemini-3.8-live-extended-thinking",
    ]:
        provider = create_llm_provider(
            provider="gemini",
            model=model_name,
            api_key="mock-key",
            use_realtime=True,
        )
        assert isinstance(provider, realtime.RealtimeModel)
        assert provider.model == model_name


# ==============================================================================
# Group 2: Realtime Connection Failure & Fallback Pipeline
# ==============================================================================

@pytest.mark.asyncio
async def test_agent_session_cascaded_pipeline_components():
    """Verify that the cascaded fallback pipeline has Silero VAD, FasterWhisperSTT, google.LLM, and bilingual TTS."""
    session = create_agent_session(
        gemini_api_key="mock-key",
        llm_model="gemini-2.5-flash",
        use_realtime=False,
    )
    assert isinstance(session, AgentSession)
    assert session.vad is not None
    assert isinstance(session.stt, stt.StreamAdapter)
    assert isinstance(session.llm, google.LLM)
    assert session.llm.model == "gemini-2.5-flash"
    assert session.tts is not None


@pytest.mark.asyncio
async def test_entrypoint_recovers_from_realtime_error_1008(monkeypatch):
    """Verify that entrypoint catches realtime model rejection (error 1008) and falls back to cascaded pipeline."""
    from zcloud_ai_telecaller import agent

    mock_room = MagicMock()
    mock_room.name = "test-recovery-room"
    mock_job = MagicMock()
    mock_job.id = "test-recovery-job"
    mock_participant = MagicMock()
    mock_participant.identity = "test-caller"
    mock_participant.attributes = {}

    mock_ctx = MagicMock()
    mock_ctx.room = mock_room
    mock_ctx.job = mock_job
    mock_ctx.proc = MagicMock()
    mock_ctx.proc.userdata = {}
    mock_ctx.connect = AsyncMock()
    mock_ctx.wait_for_participant = AsyncMock(return_value=mock_participant)
    mock_ctx.add_shutdown_callback = MagicMock()

    realtime_session_mock = MagicMock()
    realtime_session_mock.on = MagicMock(return_value=lambda fn: fn)
    # Simulate Gemini Live API error 1008 on start
    realtime_session_mock.start = AsyncMock(
        side_effect=Exception("APIError: 1008 None. models/gemini-2.5-flash is not supported for bidiGenerateContent")
    )

    fallback_session_mock = MagicMock()
    fallback_session_mock.on = MagicMock(return_value=lambda fn: fn)
    fallback_session_mock.start = AsyncMock()
    fallback_session_mock.say = AsyncMock()

    sessions_created = []

    def mock_create_agent_session(**kwargs):
        if kwargs.get("use_realtime"):
            sessions_created.append(("realtime", kwargs))
            return realtime_session_mock
        else:
            sessions_created.append(("cascaded_fallback", kwargs))
            return fallback_session_mock

    monkeypatch.setattr(agent, "create_agent_session", mock_create_agent_session)

    await agent.entrypoint(mock_ctx)

    # Verify both sessions were created (initial realtime attempt, followed by cascaded fallback)
    assert len(sessions_created) == 2
    assert sessions_created[0][0] == "realtime"
    assert sessions_created[1][0] == "cascaded_fallback"
    assert sessions_created[1][1]["llm_model"] == "gemini-2.5-flash"
    assert sessions_created[1][1]["use_realtime"] is False

    # Verify fallback session was started and greeting was spoken
    fallback_session_mock.start.assert_awaited_once()
    fallback_session_mock.say.assert_awaited_once_with(agent.GREETING_MESSAGE)


# ==============================================================================
# Group 3: Startup Greeting and Turn Count Isolation
# ==============================================================================

def test_startup_greeting_does_not_count_as_customer_turn():
    """Verify that initial agent greeting does not increment turn_count in ConversationStateManager."""
    mgr = ConversationStateManager(tenant_id="test", agent_id="agent")
    assert mgr.turn_count == 0

    # Initial assistant greeting with is_greeting=True
    mgr.add_message("assistant", "Hello! How can I help you today?", is_greeting=True)
    assert mgr.turn_count == 0  # Still 0! Greeting does not count as conversation turn
    assert len(mgr.messages) == 1

    # Caller speaks turn 1
    mgr.add_message("user", "I would like to book a dental checkup.")
    assert mgr.turn_count == 1

    # Assistant replies
    mgr.add_message("assistant", "Sure, what date works best for you?")
    assert mgr.turn_count == 2


def test_accumulator_customer_turns_and_failure_status():
    """Verify that CallSessionAccumulator distinguishes customer turns and marks abnormal exit as FAILED."""
    accum = CallSessionAccumulator(
        tenant_id="test-tenant",
        agent_id="test-agent",
        room_name="test-room",
    )
    # Only greeting played
    accum.add_transcript(speaker="AGENT", message="Hello! How can I help you today?")
    assert accum.customer_turns == 0

    # If session crashes with realtime_model_error
    report_failed = accum.finish(reason="realtime_model_error")
    assert report_failed.status == "FAILED"
    assert report_failed.end_reason == "realtime_model_error"

    # Reset accumulator with an actual conversation
    accum2 = CallSessionAccumulator(
        tenant_id="test-tenant",
        agent_id="test-agent",
        room_name="test-room",
    )
    accum2.add_transcript(speaker="AGENT", message="Hello! How can I help you today?")
    accum2.add_transcript(speaker="CUSTOMER", message="I want an appointment.")
    accum2.add_transcript(speaker="AGENT", message="Sure, what time?")
    assert accum2.customer_turns == 1

    report_success = accum2.finish(reason="normal")
    assert report_success.status == "COMPLETED"
    assert report_success.end_reason == "normal"


# ==============================================================================
# Group 4: Session Continuity and Multilingual Speech
# ==============================================================================

def test_language_preference_detection_and_switching():
    """Verify detection and continuity for Tamil, Tanglish, and English."""
    mgr = ConversationStateManager(tenant_id="test", agent_id="agent")

    # Initial preference
    assert mgr.preferred_language == "en"

    # 1. Native Tamil input
    lang = mgr.update_language_preference_from_input("வணக்கம், டாக்டர் இருக்காரா?")
    assert lang == "ta"
    assert mgr.preferred_language == "ta"

    # 2. Tanglish input
    lang = mgr.update_language_preference_from_input("Aama, root canal treatment pannanum, available timing sollunga.")
    assert lang == "thanglish"
    assert mgr.preferred_language == "thanglish"

    # 3. Explicit switch to English
    lang = mgr.update_language_preference_from_input("Can you speak in English please?")
    assert lang == "en"
    assert mgr.preferred_language == "en"

    # 4. Continuation in English maintains preference
    mgr.add_message("user", "What are the clinic opening hours?")
    assert mgr.preferred_language == "en"
    assert mgr.turn_count == 1

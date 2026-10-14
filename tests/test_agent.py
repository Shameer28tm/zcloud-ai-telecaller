"""Tests for LiveKit agent initialization, local STT/TTS pipeline, and lifecycle."""

import pytest
from livekit.agents import stt
from livekit.plugins import google

from zcloud_ai_telecaller.agent import (
    AGENT_INSTRUCTIONS,
    GREETING_MESSAGE,
    create_agent,
    create_agent_session,
    prewarm,
)
from zcloud_ai_telecaller.local_stt import FasterWhisperSTT
from zcloud_ai_telecaller.local_tts import AdaptiveBilingualTTS, PiperTTS


def test_create_agent():
    """Verify that create_agent creates a valid Agent instance with Phase 1 instructions."""
    agent = create_agent()
    assert agent is not None
    assert "You are Z Cloud AI Voice Assistant." in agent.instructions
    assert "friendly, concise and professional" in agent.instructions
    assert "Listen carefully to the caller." in agent.instructions
    assert "Answer naturally." in agent.instructions
    assert "Phase 1 technical demo" in agent.instructions
    assert agent.instructions == AGENT_INSTRUCTIONS
    assert "Hello!" in GREETING_MESSAGE


@pytest.mark.asyncio
async def test_create_agent_session():
    """Verify that create_agent_session initializes the voice pipeline with local STT/TTS and Google Gemini LLM."""
    session = create_agent_session(gemini_api_key="test-gemini-key")
    assert session is not None
    assert session.vad is not None
    assert session.stt is not None
    assert session.llm is not None
    assert session.tts is not None

    # Verify local STT is StreamAdapter wrapping FasterWhisperSTT
    assert isinstance(session.stt, stt.StreamAdapter)
    assert isinstance(session.stt.wrapped_stt, FasterWhisperSTT)
    assert session.stt.capabilities.streaming is True

    # Verify Gemini LLM is configured
    assert isinstance(session.llm, google.LLM)
    assert session.llm.model == "gemini-2.5-flash"

    # Verify local TTS is PiperTTS or AdaptiveBilingualTTS
    assert isinstance(session.tts, (PiperTTS, AdaptiveBilingualTTS))
    assert "piper-" in session.tts.model
    assert session.tts.sample_rate == 22050


def test_local_stt_capabilities():
    """Verify FasterWhisperSTT initialization and capabilities."""
    local_stt = FasterWhisperSTT(model_size_or_path="base", device="cpu", compute_type="int8")
    assert local_stt.model == "faster-whisper-base"
    assert local_stt.provider == "faster-whisper"
    assert local_stt.capabilities.streaming is False
    assert local_stt.capabilities.interim_results is False


def test_local_tts_capabilities():
    """Verify PiperTTS initialization and capabilities."""
    local_tts = PiperTTS(voice_name="en_US-lessac-medium")
    assert local_tts.model == "piper-en_US-lessac-medium"
    assert local_tts.provider == "piper"
    assert local_tts.capabilities.streaming is False
    assert local_tts.sample_rate == 22050


def test_prewarm():
    """Verify that prewarm loads Silero VAD into the JobProcess userdata."""
    class MockJobProcess:
        def __init__(self):
            self.userdata = {}

    proc = MockJobProcess()
    prewarm(proc)
    assert "vad" in proc.userdata
    assert proc.userdata["vad"] is not None


@pytest.mark.asyncio
async def test_entrypoint_lifecycle(monkeypatch):
    """Verify that entrypoint awaits session.start and registers an async shutdown callback."""
    import inspect
    from unittest.mock import AsyncMock, MagicMock
    from zcloud_ai_telecaller import agent

    mock_room = MagicMock()
    mock_room.name = "test-room"

    mock_job = MagicMock()
    mock_job.id = "test-job-id"

    mock_participant = MagicMock()
    mock_participant.identity = "test-participant"
    mock_participant.sid = "PA_123"

    registered_shutdown_callbacks = []

    mock_ctx = MagicMock()
    mock_ctx.room = mock_room
    mock_ctx.job = mock_job
    mock_ctx.proc = MagicMock()
    mock_ctx.proc.userdata = {}
    mock_ctx.connect = AsyncMock()
    mock_ctx.wait_for_participant = AsyncMock(return_value=mock_participant)
    mock_ctx.add_shutdown_callback = lambda cb: registered_shutdown_callbacks.append(cb)

    mock_session = MagicMock()
    mock_session.on = MagicMock(return_value=lambda fn: fn)
    mock_session.start = AsyncMock()
    mock_session.say = AsyncMock()

    monkeypatch.setattr(agent, "create_agent_session", lambda **kwargs: mock_session)

    await agent.entrypoint(mock_ctx)

    # 1. Verify connect and wait_for_participant were awaited
    mock_ctx.connect.assert_awaited_once()
    mock_ctx.wait_for_participant.assert_awaited_once()

    # 2. Verify session.start was awaited
    mock_session.start.assert_awaited_once()

    # 3. Verify session.say was awaited with greeting
    mock_session.say.assert_awaited_once_with(agent.GREETING_MESSAGE)

    # 4. Verify shutdown callback was registered as an async coroutine function
    assert len(registered_shutdown_callbacks) == 1
    shutdown_cb = registered_shutdown_callbacks[0]
    assert inspect.iscoroutinefunction(shutdown_cb), "Shutdown callback must be an async coroutine function"

    # 5. Verify invoking shutdown callback produces an awaitable coroutine
    await shutdown_cb()

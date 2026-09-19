"""Tests for LiveKit agent initialization and structure."""

import pytest
from zcloud_ai_telecaller.agent import (
    AGENT_INSTRUCTIONS,
    GREETING_MESSAGE,
    create_agent,
    create_agent_session,
    prewarm,
)


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
    """Verify that create_agent_session initializes the voice pipeline with Google Gemini LLM."""
    from livekit.plugins import google, openai

    session = create_agent_session(gemini_api_key="test-gemini-key")
    assert session is not None
    assert session.vad is not None
    assert session.stt is not None
    assert session.llm is not None
    assert session.tts is not None

    # Ensure Gemini LLM is configured
    assert isinstance(session.llm, google.LLM)
    assert session.llm.model == "gemini-2.5-flash"

    # Ensure OpenAI LLM is NOT instantiated in the session
    assert not isinstance(session.llm, openai.LLM)



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



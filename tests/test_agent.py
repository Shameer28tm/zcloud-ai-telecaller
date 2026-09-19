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
    """Verify that create_agent_session initializes the voice pipeline plugins (VAD, STT, LLM, TTS)."""
    session = create_agent_session()
    assert session is not None
    assert session.vad is not None
    assert session.stt is not None
    assert session.llm is not None
    assert session.tts is not None


def test_prewarm():
    """Verify that prewarm loads Silero VAD into the JobProcess userdata."""
    class MockJobProcess:
        def __init__(self):
            self.userdata = {}

    proc = MockJobProcess()
    prewarm(proc)
    assert "vad" in proc.userdata
    assert proc.userdata["vad"] is not None


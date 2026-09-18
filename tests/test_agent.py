"""Tests for LiveKit agent initialization and structure."""

from zcloud_ai_telecaller.agent import create_agent, create_agent_session


def test_create_agent():
    """Verify that create_agent creates a valid Agent instance with instructions."""
    agent = create_agent()
    assert agent is not None
    assert "Z Cloud AI Telecaller" in agent.instructions


def test_create_agent_session():
    """Verify that create_agent_session initializes the voice pipeline plugins (VAD, STT, LLM, TTS)."""
    session = create_agent_session()
    assert session is not None
    assert session.vad is not None
    assert session.stt is not None
    assert session.llm is not None
    assert session.tts is not None

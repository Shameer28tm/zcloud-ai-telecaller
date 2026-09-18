"""Tests for configuration settings and environment loading."""

import os
from zcloud_ai_telecaller.config import Settings, get_settings


def test_default_settings():
    """Verify default settings initialization."""
    settings = Settings()
    assert settings.agent_name == "zcloud-ai-telecaller-agent"
    assert settings.agent_log_level == "INFO"


def test_custom_settings_from_env(monkeypatch):
    """Verify settings pick up environment variables properly."""
    monkeypatch.setenv("AGENT_NAME", "custom-telecaller-agent")
    monkeypatch.setenv("AGENT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LIVEKIT_URL", "wss://custom.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-api-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-api-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    settings = get_settings()
    assert settings.agent_name == "custom-telecaller-agent"
    assert settings.agent_log_level == "DEBUG"
    assert settings.livekit_url == "wss://custom.livekit.cloud"
    assert settings.livekit_api_key == "test-api-key"
    assert settings.livekit_api_secret == "test-api-secret"
    assert settings.openai_api_key == "test-openai-key"

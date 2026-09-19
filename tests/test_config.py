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
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    settings = get_settings()
    assert settings.agent_name == "custom-telecaller-agent"
    assert settings.agent_log_level == "DEBUG"
    assert settings.livekit_url == "wss://custom.livekit.cloud"
    assert settings.livekit_api_key == "test-api-key"
    assert settings.livekit_api_secret == "test-api-secret"
    assert settings.gemini_api_key == "test-gemini-key"
    assert settings.openai_api_key == "test-openai-key"
    assert settings.is_livekit_configured is True
    assert settings.is_gemini_configured is True
    assert settings.is_openai_configured is True


def test_validate_environment_success(monkeypatch):
    """Verify validate_environment passes when all required variables are set."""
    monkeypatch.setenv("LIVEKIT_URL", "wss://test.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key123")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret123")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestKey123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test12345")

    settings = Settings()
    # Should complete without error
    settings.validate_environment()


def test_validate_environment_missing_keys(monkeypatch):
    """Verify validate_environment raises ValueError listing missing keys."""
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = Settings(
        livekit_url=None,
        livekit_api_key=None,
        livekit_api_secret=None,
        gemini_api_key=None,
        openai_api_key=None,
    )
    import pytest

    with pytest.raises(ValueError) as exc_info:
        settings.validate_environment()

    err_msg = str(exc_info.value)
    assert "LIVEKIT_URL" in err_msg
    assert "LIVEKIT_API_KEY" in err_msg
    assert "LIVEKIT_API_SECRET" in err_msg
    assert "GEMINI_API_KEY" in err_msg
    assert "OPENAI_API_KEY" in err_msg


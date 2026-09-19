"""Tests for configuration settings and environment loading."""

import pytest
from zcloud_ai_telecaller.config import Settings, get_settings


def test_default_settings():
    """Verify default settings initialization."""
    settings = Settings()
    assert settings.agent_name == "zcloud-ai-telecaller-agent"
    assert settings.agent_log_level == "INFO"
    assert settings.whisper_model_size == "base"
    assert settings.whisper_device == "cpu"
    assert settings.whisper_compute_type == "int8"
    assert settings.piper_voice == "en_US-lessac-medium"
    assert settings.piper_model_path is None
    assert settings.piper_config_path is None


def test_custom_settings_from_env(monkeypatch):
    """Verify settings pick up environment variables properly."""
    monkeypatch.setenv("AGENT_NAME", "custom-telecaller-agent")
    monkeypatch.setenv("AGENT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LIVEKIT_URL", "wss://custom.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-api-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-api-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "tiny")
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("PIPER_VOICE", "en_US-lessac-low")
    monkeypatch.setenv("PIPER_MODEL_PATH", "/path/to/custom.onnx")
    monkeypatch.setenv("PIPER_CONFIG_PATH", "/path/to/custom.onnx.json")

    settings = get_settings()
    assert settings.agent_name == "custom-telecaller-agent"
    assert settings.agent_log_level == "DEBUG"
    assert settings.livekit_url == "wss://custom.livekit.cloud"
    assert settings.livekit_api_key == "test-api-key"
    assert settings.livekit_api_secret == "test-api-secret"
    assert settings.gemini_api_key == "test-gemini-key"
    assert settings.whisper_model_size == "tiny"
    assert settings.whisper_device == "cpu"
    assert settings.whisper_compute_type == "int8"
    assert settings.piper_voice == "en_US-lessac-low"
    assert settings.piper_model_path == "/path/to/custom.onnx"
    assert settings.piper_config_path == "/path/to/custom.onnx.json"
    assert settings.is_livekit_configured is True
    assert settings.is_gemini_configured is True


def test_control_plane_configuration(monkeypatch):
    """Verify control plane URL and worker API secret loading and property flag."""
    settings_default = Settings(control_plane_url="http://localhost:3000", worker_api_secret=None)
    assert settings_default.control_plane_url == "http://localhost:3000"
    assert settings_default.worker_api_secret is None
    assert settings_default.is_control_plane_configured is False

    monkeypatch.setenv("CONTROL_PLANE_URL", "https://controlplane.zcloud.internal")
    monkeypatch.setenv("WORKER_API_SECRET", "super_secret_m2m_token_99")
    settings_custom = Settings()
    assert settings_custom.control_plane_url == "https://controlplane.zcloud.internal"
    assert settings_custom.worker_api_secret == "super_secret_m2m_token_99"
    assert settings_custom.is_control_plane_configured is True


def test_validate_environment_success_without_openai(monkeypatch):
    """Verify validate_environment passes without any OPENAI_API_KEY set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "wss://test.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key123")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret123")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestKey123")

    settings = Settings()
    # Must complete without error, requiring zero OpenAI configuration
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
    )

    with pytest.raises(ValueError) as exc_info:
        settings.validate_environment()

    err_msg = str(exc_info.value)
    assert "LIVEKIT_URL" in err_msg
    assert "LIVEKIT_API_KEY" in err_msg
    assert "LIVEKIT_API_SECRET" in err_msg
    assert "GEMINI_API_KEY" in err_msg
    assert "OPENAI_API_KEY" not in err_msg

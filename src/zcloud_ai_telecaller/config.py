"""Configuration management for Z Cloud AI Telecaller using Pydantic Settings."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LiveKit credentials
    livekit_url: Optional[str] = Field(
        default=None,
        validation_alias="LIVEKIT_URL",
        description="LiveKit WebRTC Server URL (e.g. wss://...)",
    )
    livekit_api_key: Optional[str] = Field(
        default=None,
        validation_alias="LIVEKIT_API_KEY",
        description="LiveKit API Key",
    )
    livekit_api_secret: Optional[str] = Field(
        default=None,
        validation_alias="LIVEKIT_API_SECRET",
        description="LiveKit API Secret",
    )

    # Google Gemini API key (for LLM reasoning & responses)
    gemini_api_key: Optional[str] = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
        description="Google Gemini API Key for LLM plugin",
    )

    # Local STT (faster-whisper) settings
    whisper_model_size: str = Field(
        default="base",
        validation_alias="WHISPER_MODEL_SIZE",
        description="Whisper model size (e.g. tiny, base, small, medium, large-v3)",
    )
    whisper_device: str = Field(
        default="cpu",
        validation_alias="WHISPER_DEVICE",
        description="Whisper compute device ('cpu', 'cuda', 'auto')",
    )
    whisper_compute_type: str = Field(
        default="int8",
        validation_alias="WHISPER_COMPUTE_TYPE",
        description="Whisper quantization/compute type (e.g. 'int8', 'float16', 'default')",
    )

    # Local TTS (Piper) settings
    piper_voice: str = Field(
        default="en_US-lessac-medium",
        validation_alias="PIPER_VOICE",
        description="Piper voice model name (e.g. 'en_US-lessac-medium', 'en_US-lessac-low')",
    )
    piper_model_path: Optional[str] = Field(
        default=None,
        validation_alias="PIPER_MODEL_PATH",
        description="Direct path to Piper .onnx model file (optional)",
    )
    piper_config_path: Optional[str] = Field(
        default=None,
        validation_alias="PIPER_CONFIG_PATH",
        description="Direct path to Piper .onnx.json config file (optional)",
    )

    # Agent runtime settings
    agent_name: str = Field(
        default="zcloud-ai-telecaller-agent",
        validation_alias="AGENT_NAME",
        description="Agent worker name",
    )
    agent_log_level: str = Field(
        default="INFO",
        validation_alias="AGENT_LOG_LEVEL",
        description="Agent logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    @property
    def is_livekit_configured(self) -> bool:
        """Check if all LiveKit connection credentials are provided."""
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)

    @property
    def is_gemini_configured(self) -> bool:
        """Check if Google Gemini API key is provided."""
        return bool(self.gemini_api_key)

    def validate_environment(self) -> None:
        """Validates that all required environment variables for LiveKit and Gemini LLM are set.

        Raises:
            ValueError: If any required configuration variables are missing or empty.
        """
        missing: list[str] = []
        if not self.livekit_url:
            missing.append("LIVEKIT_URL")
        if not self.livekit_api_key:
            missing.append("LIVEKIT_API_KEY")
        if not self.livekit_api_secret:
            missing.append("LIVEKIT_API_SECRET")
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")

        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Please configure them in your .env file or host environment."
            )



def get_settings() -> Settings:
    """Instantiate and return the application settings."""
    return Settings()

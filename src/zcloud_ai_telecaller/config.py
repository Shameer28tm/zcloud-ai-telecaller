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

    # OpenAI API key (retained for STT and TTS plugins)
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API Key for STT (Whisper) and TTS (Alloy) plugins",
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

    @property
    def is_openai_configured(self) -> bool:
        """Check if OpenAI API key is provided."""
        return bool(self.openai_api_key)

    def validate_environment(self) -> None:
        """Validates that all required environment variables for LiveKit, Gemini LLM, and OpenAI STT/TTS are set.

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
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY (required for OpenAI STT/TTS)")

        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Please configure them in your .env file or host environment."
            )



def get_settings() -> Settings:
    """Instantiate and return the application settings."""
    return Settings()

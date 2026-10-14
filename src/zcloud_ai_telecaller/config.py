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
    livekit_agent_deployment: str = Field(
        default="",
        validation_alias="LIVEKIT_AGENT_DEPLOYMENT",
        description="LiveKit Agent Deployment environment (e.g. 'production' or '')",
    )

    # Google Gemini API key (for LLM reasoning & responses)
    gemini_api_key: Optional[str] = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
        description="Google Gemini API Key for LLM plugin",
    )
    gemini_model: str = Field(
        default="gemini-3.8-live",
        validation_alias="GEMINI_MODEL",
        description="Google Gemini model identifier (e.g. 'gemini-3.8-live', 'gemini-2.5-flash')",
    )
    use_realtime_audio: bool = Field(
        default=True,
        validation_alias="USE_REALTIME_AUDIO",
        description="Enable native real-time audio-to-audio streaming via Gemini Live API",
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
    whisper_language: Optional[str] = Field(
        default=None,
        validation_alias="WHISPER_LANGUAGE",
        description="Whisper language code ('en', 'ta', or None for auto-detect bilingual)",
    )
    whisper_initial_prompt: Optional[str] = Field(
        default="Vanakkam, Tamil, தமிழ், Thanglish clinic appointment booking: dental cleaning, consultation, doctor, receptionist. Enakku appointment book pannanum.",
        validation_alias="WHISPER_INITIAL_PROMPT",
        description="Initial prompt for Faster-Whisper to prime bilingual English, Tamil, and Thanglish recognition",
    )

    # Local TTS (Piper) and Cloud Indic TTS (Sarvam Bulbul) settings
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
    sarvam_api_key: Optional[str] = Field(
        default=None,
        validation_alias="SARVAM_API_KEY",
        description="Optional API key for Sarvam AI Bulbul Tamil TTS",
    )
    sarvam_model: str = Field(
        default="bulbul:v3",
        validation_alias="SARVAM_MODEL",
        description="Sarvam TTS model name (default: bulbul:v3)",
    )

    # LLM Provider and Local AI settings
    llm_provider: str = Field(
        default="gemini",
        validation_alias="LLM_PROVIDER",
        description="Active LLM provider: 'gemini', 'ollama', or 'hybrid'",
    )
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias="OLLAMA_BASE_URL",
        description="Base URL for local Ollama HTTP API",
    )
    ollama_model: str = Field(
        default="qwen3:4b",
        validation_alias="OLLAMA_MODEL",
        description="Local Ollama model identifier (e.g. 'qwen3:4b', 'llama3')",
    )
    ollama_timeout: float = Field(
        default=60.0,
        validation_alias="OLLAMA_TIMEOUT",
        description="Timeout in seconds for Ollama inference requests",
    )
    ai_mode: str = Field(
        default="CLOUD_AI",
        validation_alias="AI_MODE",
        description="Operating AI mode: 'LOCAL_AI', 'CLOUD_AI', or 'HYBRID'",
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

    # Control Plane Runtime API settings (Phase 2A)
    control_plane_url: str = Field(
        default="http://localhost:3000",
        validation_alias="CONTROL_PLANE_URL",
        description="Base URL for the Next.js control plane (e.g. http://localhost:3000)",
    )
    worker_api_secret: Optional[str] = Field(
        default=None,
        validation_alias="WORKER_API_SECRET",
        description="Internal machine-to-machine worker API secret for control plane authentication",
    )

    # Phase 4 Telephony & LiveKit SIP settings
    livekit_sip_uri: str = Field(
        default="sip.livekit.cloud",
        validation_alias="LIVEKIT_SIP_URI",
        description="LiveKit SIP URI domain (e.g. sip.livekit.cloud)",
    )
    sip_simulation_mode: bool = Field(
        default=True,
        validation_alias="SIP_SIMULATION_MODE",
        description="Run SIP telephony in simulation mode when real provider credentials are not yet provisioned",
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
    def is_control_plane_configured(self) -> bool:
        """Check if control plane connection and secret are configured."""
        return bool(self.control_plane_url and self.worker_api_secret)

    def validate_environment(self) -> None:
        """Validates that all required environment variables for the selected AI mode and LiveKit are set.

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

        is_local_mode = (
            self.llm_provider.lower() in ("ollama", "local")
            or self.ai_mode.upper() == "LOCAL_AI"
        )
        if not is_local_mode and not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")

        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Please configure them in your .env file or host environment."
            )



def get_settings() -> Settings:
    """Instantiate and return the application settings."""
    return Settings()

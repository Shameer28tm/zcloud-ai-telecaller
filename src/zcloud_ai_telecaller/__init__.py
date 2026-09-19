"""Z Cloud AI Telecaller - LiveKit AI Voice Agent Foundation."""

from zcloud_ai_telecaller.config import Settings, get_settings
from zcloud_ai_telecaller.logging_config import setup_logging


from zcloud_ai_telecaller.local_stt import FasterWhisperSTT
from zcloud_ai_telecaller.local_tts import PiperTTS
from zcloud_ai_telecaller.runtime_client import (
    AgentRuntimeConfig,
    RuntimeConfigClient,
    extract_participant_metadata,
)


def main() -> None:
    """Console script entry point that runs the LiveKit agent."""
    from zcloud_ai_telecaller.agent import main as run_agent

    run_agent()


__all__ = [
    "main",
    "Settings",
    "get_settings",
    "setup_logging",
    "FasterWhisperSTT",
    "PiperTTS",
    "AgentRuntimeConfig",
    "RuntimeConfigClient",
    "extract_participant_metadata",
]


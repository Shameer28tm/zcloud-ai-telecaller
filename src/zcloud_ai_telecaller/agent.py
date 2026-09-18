"""LiveKit AI Voice Agent entry point for Z Cloud AI Telecaller.

Voice Pipeline:
Browser/WebRTC -> LiveKit Room -> Silero VAD / OpenAI STT -> OpenAI LLM -> OpenAI TTS -> LiveKit Room -> Browser
"""

import asyncio
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import openai, silero

from zcloud_ai_telecaller.config import get_settings
from zcloud_ai_telecaller.logging_config import setup_logging


def create_agent() -> Agent:
    """Creates the base AI voice agent configured with conversational instructions."""
    return Agent(
        instructions=(
            "You are the Z Cloud AI Telecaller assistant. "
            "Your responses should be friendly, clear, concise, and conversational. "
            "You are speaking directly with a user over an audio call."
        )
    )


def create_agent_session() -> AgentSession:
    """Creates and configures the AgentSession with VAD, STT, LLM, and TTS plugins."""
    return AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(model="tts-1", voice="alloy"),
    )


async def entrypoint(ctx: JobContext) -> None:
    """Main job entry point triggered whenever an agent worker is assigned to a LiveKit room."""
    settings = get_settings()
    logger = setup_logging(settings.agent_log_level)

    logger.info("Connecting to LiveKit room: %s", ctx.room.name)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Initialize the voice pipeline session and agent
    session = create_agent_session()
    agent = create_agent()

    logger.info("Starting AgentSession for room: %s", ctx.room.name)
    session.start(agent, room=ctx.room)

    # Greet user once connected
    await session.say("Hello! I am your Z Cloud AI Telecaller assistant. How can I help you today?")


def main() -> None:
    """Entry point to run the LiveKit Agent worker."""
    settings = get_settings()
    setup_logging(settings.agent_log_level)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()

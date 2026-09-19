"""LiveKit AI Voice Agent entry point for Z Cloud AI Telecaller.

Voice Pipeline:
Browser/WebRTC -> LiveKit Room -> Silero VAD / local faster-whisper STT -> Gemini 2.5 Flash LLM -> local Piper TTS -> LiveKit Room -> Browser
"""

import asyncio
import logging
import os
import sys
from typing import Optional

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    stt,
)
from livekit.plugins import google, silero

from zcloud_ai_telecaller.config import Settings, get_settings
from zcloud_ai_telecaller.local_stt import FasterWhisperSTT
from zcloud_ai_telecaller.local_tts import PiperTTS
from zcloud_ai_telecaller.logging_config import setup_logging
from zcloud_ai_telecaller.runtime_client import (
    AgentRuntimeConfig,
    RuntimeConfigClient,
    extract_participant_metadata,
)

AGENT_INSTRUCTIONS = (
    "You are Z Cloud AI Voice Assistant.\n"
    "You are friendly, concise and professional.\n"
    "Listen carefully to the caller.\n"
    "Answer naturally.\n"
    "If you do not know something, say so instead of inventing information.\n"
    "This is only a Phase 1 technical demo."
)

GREETING_MESSAGE = "Hello! I am your Z Cloud AI Voice Assistant. How can I help you today?"


def create_agent(instructions: str = AGENT_INSTRUCTIONS) -> Agent:
    """Creates the base AI voice agent configured with Phase 1 conversational instructions."""
    return Agent(instructions=instructions)


def create_agent_session(
    vad: Optional[silero.VAD] = None,
    gemini_api_key: Optional[str] = None,
    whisper_model_size: str = "base",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    piper_voice: str = "en_US-lessac-medium",
    piper_model_path: Optional[str] = None,
    piper_config_path: Optional[str] = None,
) -> AgentSession:
    """Creates and configures the AgentSession with Silero VAD, local faster-whisper STT, Google Gemini LLM, and local Piper TTS."""
    # Resolve Gemini API key for Google LLM
    effective_gemini_key = (
        gemini_api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if effective_gemini_key:
        os.environ["GOOGLE_API_KEY"] = effective_gemini_key

    vad_instance = vad or silero.VAD.load()

    # Local STT using faster-whisper adapted for streaming audio
    local_whisper = FasterWhisperSTT(
        model_size_or_path=whisper_model_size,
        device=whisper_device,
        compute_type=whisper_compute_type,
    )
    stt_stream = stt.StreamAdapter(stt=local_whisper, vad=vad_instance)

    # Local TTS using Piper neural voice
    local_piper = PiperTTS(
        voice_name=piper_voice,
        model_path=piper_model_path,
        config_path=piper_config_path,
    )

    return AgentSession(
        vad=vad_instance,
        stt=stt_stream,
        llm=google.LLM(
            model="gemini-2.5-flash",
            api_key=effective_gemini_key or "dummy-key-for-initialization",
        ),
        tts=local_piper,
    )


def prewarm(proc: JobProcess) -> None:
    """Pre-warm models during worker startup for low-latency initial responses."""
    logger = logging.getLogger("zcloud_ai_telecaller")
    logger.info("Pre-warming Silero VAD model in worker process...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Worker process pre-warm complete")


async def entrypoint(ctx: JobContext) -> None:
    """Main job entry point triggered whenever an agent worker is assigned to a LiveKit room."""
    settings = get_settings()
    logger = setup_logging(settings.agent_log_level)

    logger.info("Worker job received for room: %s (job_id: %s)", ctx.room.name, ctx.job.id)

    # 1. Connect to LiveKit room
    try:
        logger.info("Connecting to LiveKit room: %s", ctx.room.name)
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info("LiveKit connection established for room: %s", ctx.room.name)
    except Exception as exc:
        logger.error("Failed to connect to LiveKit room %s: %s", ctx.room.name, exc)
        raise

    # 2. Wait for a participant to join the room
    logger.info("Waiting for participant to join room: %s...", ctx.room.name)
    try:
        participant = await ctx.wait_for_participant()
        logger.info(
            "Participant joined: identity=%s, sid=%s in room: %s",
            participant.identity,
            participant.sid,
            ctx.room.name,
        )
    except Exception as exc:
        logger.error("Error waiting for participant in room %s: %s", ctx.room.name, exc)
        raise

    # 3. Resolve dynamic agent configuration from control plane
    instructions = AGENT_INSTRUCTIONS
    greeting = GREETING_MESSAGE
    piper_voice = settings.piper_voice

    metadata_raw: Optional[str] = None
    part_meta = getattr(participant, "metadata", None)
    room_meta = getattr(ctx.room, "metadata", None)
    if isinstance(part_meta, str) and part_meta.strip():
        metadata_raw = part_meta
    elif isinstance(room_meta, str) and room_meta.strip():
        metadata_raw = room_meta

    if metadata_raw:
        try:
            tenant_id, agent_id, agent_version_id = extract_participant_metadata(metadata_raw)
            logger.info(
                "Participant metadata resolved: tenant_id=%s, agent_id=%s, version_id=%s",
                tenant_id,
                agent_id,
                agent_version_id,
            )
            client = RuntimeConfigClient(
                base_url=settings.control_plane_url,
                worker_api_secret=settings.worker_api_secret,
            )
            runtime_config = await client.fetch_agent_runtime(
                tenant_id=tenant_id,
                agent_id=agent_id,
                agent_version_id=agent_version_id,
            )
            logger.info(
                "Dynamic runtime config successfully loaded for agent '%s' (version=%s)",
                runtime_config.name,
                runtime_config.agent_version_id,
            )
            instructions = runtime_config.prompt
            greeting = runtime_config.greeting
            if runtime_config.voice_id:
                piper_voice = runtime_config.voice_id
        except Exception as exc:
            logger.error(
                "Failed to fetch or apply runtime configuration from control plane: %s. Aborting session.",
                exc,
            )
            return
    elif settings.is_control_plane_configured and not os.environ.get("PYTEST_CURRENT_TEST"):
        logger.error(
            "Participant metadata missing in room %s while control plane is configured. Aborting session.",
            ctx.room.name,
        )
        return

    # 4. Retrieve pre-warmed VAD if available, or load fresh
    vad = ctx.proc.userdata.get("vad")

    # 5. Initialize voice pipeline session and agent with resolved configuration
    try:
        session = create_agent_session(
            vad=vad,
            gemini_api_key=settings.gemini_api_key,
            whisper_model_size=settings.whisper_model_size,
            whisper_device=settings.whisper_device,
            whisper_compute_type=settings.whisper_compute_type,
            piper_voice=piper_voice,
            piper_model_path=settings.piper_model_path,
            piper_config_path=settings.piper_config_path,
        )
        agent = create_agent(instructions=instructions)
    except Exception as exc:
        logger.error("Failed to initialize voice session components: %s", exc)
        raise

    # 6. Attach event listeners for structured lifecycle logging & error tracking
    @session.on("error")
    def on_session_error(ev) -> None:
        err = getattr(ev, "error", ev)
        logger.error("Session error occurred in room %s: %s", ctx.room.name, err)

    @session.on("close")
    def on_session_close(ev) -> None:
        reason = getattr(ev, "reason", "normal")
        err = getattr(ev, "error", None)
        if err:
            logger.error("Session ended with error in room %s: %s (reason: %s)", ctx.room.name, err, reason)
        else:
            logger.info("Session ended normally for room %s (reason: %s)", ctx.room.name, reason)

    async def on_shutdown() -> None:
        logger.info("Worker session cleanup/shutdown completed for room: %s", ctx.room.name)

    ctx.add_shutdown_callback(on_shutdown)

    # 7. Start the voice pipeline session
    logger.info("Voice session started for room: %s with participant: %s", ctx.room.name, participant.identity)
    await session.start(agent, room=ctx.room)

    # 8. Initial voice greeting to caller
    await session.say(greeting)


def main() -> None:
    """Entry point to run the LiveKit Agent worker."""
    settings = get_settings()
    logger = setup_logging(settings.agent_log_level)

    logger.info(
        "Worker startup: Initializing Z Cloud AI Telecaller agent worker (agent=%s)",
        settings.agent_name,
    )

    # Validate environment when starting the agent worker
    is_start_or_dev = len(sys.argv) < 2 or sys.argv[1] in ("start", "dev", "console")
    if is_start_or_dev:
        try:
            settings.validate_environment()
            logger.info("Environment configuration validated successfully")
        except ValueError as err:
            logger.error("Configuration validation failed: %s", err)
            sys.exit(1)

    # Populate API keys in environment for provider SDKs
    if settings.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
        os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            ws_url=settings.livekit_url,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            agent_name=settings.agent_name,
        )
    )


if __name__ == "__main__":
    main()


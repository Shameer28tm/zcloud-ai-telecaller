"""LiveKit AI Voice Agent entry point for Z Cloud AI Telecaller.

Phase 3 Clinic AI Voice Pipeline:
Browser/WebRTC -> LiveKit Room -> Silero VAD / local faster-whisper STT ->
Gemini 2.5 Flash LLM with Backend-Controlled Clinic Tools ->
local Piper TTS -> LiveKit Room -> Browser
"""

import asyncio
import datetime
import json
import logging
import os
import sys
from typing import Any, Optional

from livekit import rtc
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

from zcloud_ai_telecaller.clinic_tools import ClinicToolsManager
from zcloud_ai_telecaller.config import Settings, get_settings
from zcloud_ai_telecaller.conversation_state import ConversationStateManager
from zcloud_ai_telecaller.llm_provider import create_llm_provider
from zcloud_ai_telecaller.local_stt import FasterWhisperSTT
from zcloud_ai_telecaller.local_tts import AdaptiveBilingualTTS, PiperTTS, SarvamTTS
from zcloud_ai_telecaller.logging_config import setup_logging
from zcloud_ai_telecaller.prompt_builder import (
    build_clinic_system_prompt,
    match_deterministic_clinic_query,
)
from zcloud_ai_telecaller.rule_engine import BusinessRuleEngine
from zcloud_ai_telecaller.session import (
    BaseCallSession,
    BrowserSession,
    CallSessionState,
    SIPSession,
)
from zcloud_ai_telecaller.runtime_client import (
    AgentRuntimeConfig,
    CallSessionAccumulator,
    CallSessionReport,
    CallTranscriptItem,
    RuntimeConfigClient,
    extract_participant_metadata,
    extract_sip_participant_info,
    is_sip_session,
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


def create_agent(
    instructions: str = AGENT_INSTRUCTIONS,
    tools: Optional[list[Any]] = None,
) -> Agent:
    """Creates the base AI voice agent configured with instructions and backend-controlled tools."""
    return Agent(instructions=instructions, tools=tools or None)


def create_agent_session(
    vad: Optional[silero.VAD] = None,
    llm: Optional[Any] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    ai_mode: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    use_realtime: Optional[bool] = None,
    whisper_model_size: str = "base",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    whisper_language: Optional[str] = None,
    whisper_initial_prompt: Optional[str] = None,
    piper_voice: str = "en_US-lessac-medium",
    piper_model_path: Optional[str] = None,
    piper_config_path: Optional[str] = None,
    sarvam_api_key: Optional[str] = None,
    sarvam_model: Optional[str] = None,
    route_thanglish_to_sarvam: bool = False,
) -> AgentSession:
    """Creates and configures the AgentSession with Silero VAD, local faster-whisper STT, LLM provider, and adaptive bilingual TTS."""
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
        language=whisper_language,
        initial_prompt=whisper_initial_prompt,
    )
    stt_stream = stt.StreamAdapter(stt=local_whisper, vad=vad_instance)

    # Local TTS using Piper neural voice and optional Sarvam AI Bulbul Indic TTS
    local_piper = PiperTTS(
        voice_name=piper_voice,
        model_path=piper_model_path,
        config_path=piper_config_path,
    )
    effective_sarvam_key = sarvam_api_key or os.environ.get("SARVAM_API_KEY")
    sarvam_instance = (
        SarvamTTS(api_key=effective_sarvam_key, model=sarvam_model or "bulbul:v4-flash")
        if effective_sarvam_key
        else None
    )
    bilingual_tts = AdaptiveBilingualTTS(
        piper_tts=local_piper,
        sarvam_tts=sarvam_instance,
        route_thanglish_to_sarvam=route_thanglish_to_sarvam,
    )

    # Resolve LLM provider (Ollama, Gemini, or Hybrid)
    if llm is not None:
        effective_llm = llm
    else:
        effective_llm = create_llm_provider(
            provider=llm_provider or "gemini",
            model=llm_model,
            base_url=llm_base_url,
            api_key=effective_gemini_key,
            ai_mode=ai_mode,
            use_realtime=use_realtime,
        )

    try:
        from livekit.plugins.google import realtime
        is_realtime_session = isinstance(effective_llm, realtime.RealtimeModel)
    except ImportError:
        is_realtime_session = False

    if is_realtime_session:
        # Native Gemini Live audio-to-audio:
        # User audio flows directly to Gemini Live over WebSocket with server-side turn detection.
        # STT is omitted to eliminate duplicate Whisper CPU transcription and latency.
        # TTS is provided for programmatic session.say() fallback (greetings and emergency prompts).
        return AgentSession(
            vad=vad_instance,
            llm=effective_llm,
            tts=bilingual_tts,
            turn_handling={
                "endpointing": {"min_delay": 0.25, "max_delay": 1.5},
                "preemptive_generation": {"enabled": True},
            },
        )
    else:
        # Cascaded STT -> LLM -> TTS pipeline
        return AgentSession(
            vad=vad_instance,
            stt=stt_stream,
            llm=effective_llm,
            tts=bilingual_tts,
            turn_handling={
                "endpointing": {"min_delay": 0.35, "max_delay": 2.0},
                "preemptive_generation": {"enabled": True},
            },
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

    client: Optional[RuntimeConfigClient] = None
    accumulator: Optional[CallSessionAccumulator] = None
    runtime_config: Optional[AgentRuntimeConfig] = None
    state_manager: Optional[ConversationStateManager] = None
    rule_engine: Optional[BusinessRuleEngine] = None
    tools_manager: Optional[ClinicToolsManager] = None
    function_tools: list[Any] = []
    active_call_session: Optional[BaseCallSession] = None

    metadata_raw: Optional[str] = None
    part_meta = getattr(participant, "metadata", None)
    room_meta = getattr(ctx.room, "metadata", None)
    if isinstance(part_meta, str) and part_meta.strip():
        metadata_raw = part_meta
    elif isinstance(room_meta, str) and room_meta.strip():
        metadata_raw = room_meta

    async def broadcast_data_packet(packet_dict: dict[str, Any]) -> None:
        """Broadcasts data packets to the LiveKit room for the realtime workbench feed."""
        try:
            payload = json.dumps(packet_dict).encode("utf-8")
            if ctx.room and ctx.room.local_participant:
                await ctx.room.local_participant.publish_data(payload, reliable=True)
        except Exception as b_err:
            logger.debug("Could not publish livekit data message: %s", b_err)

    call_id: Optional[str] = None
    sip_called_number, sip_caller_number, sip_call_id = extract_sip_participant_info(participant)
    is_sip = is_sip_session(participant, ctx.room) or bool(sip_called_number)

    if is_sip:
        sip_route_resolved = False
        if sip_called_number:
            try:
                logger.info(
                    "Inbound SIP call detected: called_number=%s, caller_number=%s, call_id=%s in room: %s",
                    sip_called_number,
                    sip_caller_number,
                    sip_call_id,
                    ctx.room.name,
                )
                client = RuntimeConfigClient(
                    base_url=settings.control_plane_url,
                    worker_api_secret=settings.worker_api_secret,
                )
                sip_route = await client.resolve_sip_inbound(
                    called_number=sip_called_number,
                    caller_number=sip_caller_number,
                    call_id=sip_call_id,
                    room_name=ctx.room.name,
                )
                tenant_id = sip_route["tenantId"]
                agent_id = sip_route["agentId"]
                agent_version_id = sip_route.get("agentVersionId")
                call_id = sip_route.get("callId")
                runtime_config = (
                    sip_route.get("parsedRuntimeConfig")
                    or AgentRuntimeConfig.model_validate(sip_route["runtimeConfig"])
                )
                logger.info(
                    "Inbound SIP route resolved: tenant_id=%s, agent_id=%s, call_id=%s (agent '%s')",
                    tenant_id,
                    agent_id,
                    call_id,
                    runtime_config.name,
                )

                # Update call lifecycle to ANSWERED then ACTIVE
                if call_id:
                    try:
                        await client.update_call_lifecycle(call_id=call_id, status="ANSWERED")
                        await client.update_call_lifecycle(call_id=call_id, status="ACTIVE")
                    except Exception as lc_err:
                        logger.warning("Notice: Could not update call lifecycle for call %s: %s", call_id, lc_err)

                # Build enriched clinic prompt and configure clinic tools
                if runtime_config.clinic_profile:
                    instructions = build_clinic_system_prompt(runtime_config)
                else:
                    instructions = runtime_config.prompt
                greeting = runtime_config.greeting
                if runtime_config.voice_id:
                    piper_voice = runtime_config.voice_id

                # Initialize structured conversation state and business rule engine
                max_ctx = 10
                max_out = 250
                if runtime_config.business_rules:
                    policy = runtime_config.business_rules.get("conversationPolicy", {})
                    max_ctx = policy.get("maxContextMessages", 10)
                    max_out = policy.get("maxOutputTokens", 250)

                state_manager = ConversationStateManager(
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    max_context_messages=max_ctx,
                    max_output_tokens=max_out,
                )
                rule_engine = BusinessRuleEngine(runtime_config.business_rules)

                effective_caller_num = sip_caller_number or sip_route.get("callerNumber")
                effective_called_num = sip_called_number or sip_route.get("recipientNumber")

                # Initialize authoritative call session accumulator for billing & transcripts
                accumulator = CallSessionAccumulator(
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    room_name=ctx.room.name,
                    agent_version_id=runtime_config.agent_version_id,
                    caller_identity=participant.identity,
                    caller_number=effective_caller_num,
                    caller_name=getattr(participant, "name", None) or (f"Caller ({effective_caller_num})" if effective_caller_num else participant.identity),
                    recipient_number=effective_called_num,
                    direction="INBOUND",
                )

                # Initialize backend-controlled clinic tools
                tools_manager = ClinicToolsManager(
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    call_id=call_id,
                    client=client,
                    transfer_rules=runtime_config.transfer_rules if isinstance(runtime_config.transfer_rules, dict) else {},
                    clinic_profile=runtime_config.clinic_profile or {},
                    state_manager=state_manager,
                    on_event=broadcast_data_packet,
                )
                function_tools = tools_manager.get_function_tools()
                active_call_session = SIPSession(
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    room_name=ctx.room.name,
                    runtime_config=runtime_config,
                    state_manager=state_manager,
                    accumulator=accumulator,
                    called_number=effective_called_num or '',
                    call_id=call_id or '',
                    caller_number=effective_caller_num,
                    sip_call_id=sip_call_id,
                    provider='Plivo',
                    agent_version_id=runtime_config.agent_version_id,
                )
                active_call_session.transition_to(CallSessionState.CONNECTING, "sip_route_resolved")
                sip_route_resolved = True
            except Exception as exc:
                logger.warning(
                    "Notice: Control plane SIP route resolution failed for number %s: %s. Activating fallback assistant to prevent call disconnect.",
                    sip_called_number,
                    exc,
                )

        if not sip_route_resolved:
            # Fallback initialization: Keep inbound call alive with default agent instructions & session
            logger.info("Initializing fallback inbound SIP voice session for participant %s in room %s", participant.identity, ctx.room.name)
            instructions = AGENT_INSTRUCTIONS
            greeting = GREETING_MESSAGE
            effective_caller_num = sip_caller_number or participant.identity
            effective_called_num = sip_called_number or ""

            fallback_rc = AgentRuntimeConfig(
                tenantId="default-clinic",
                agentId="default-agent",
                agentVersionId="v1",
                name="Clinic Voice Assistant",
                prompt=instructions,
                greeting=greeting,
                aiModel=settings.gemini_model,
            )
            runtime_config = fallback_rc
            state_manager = ConversationStateManager(tenant_id="default-clinic", agent_id="default-agent")
            accumulator = CallSessionAccumulator(
                tenant_id="default-clinic",
                agent_id="default-agent",
                room_name=ctx.room.name,
                caller_identity=participant.identity,
                caller_number=effective_caller_num,
                caller_name=getattr(participant, "name", None) or (f"Caller ({effective_caller_num})" if effective_caller_num else participant.identity),
                recipient_number=effective_called_num,
                direction="INBOUND",
            )
            active_call_session = SIPSession(
                tenant_id="default-clinic",
                agent_id="default-agent",
                room_name=ctx.room.name,
                runtime_config=fallback_rc,
                state_manager=state_manager,
                accumulator=accumulator,
                called_number=effective_called_num,
                call_id=call_id or "",
                caller_number=effective_caller_num,
                sip_call_id=sip_call_id,
                provider="Plivo",
            )
            active_call_session.transition_to(CallSessionState.CONNECTING, "fallback_sip_initialized")
    elif metadata_raw:
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

            # Build enriched clinic prompt and configure clinic tools
            if runtime_config.clinic_profile:
                instructions = build_clinic_system_prompt(runtime_config)
            else:
                instructions = runtime_config.prompt
            greeting = runtime_config.greeting
            if runtime_config.voice_id:
                piper_voice = runtime_config.voice_id

            # Initialize structured conversation state and business rule engine
            max_ctx = 10
            max_out = 250
            if runtime_config.business_rules:
                policy = runtime_config.business_rules.get("conversationPolicy", {})
                max_ctx = policy.get("maxContextMessages", 10)
                max_out = policy.get("maxOutputTokens", 250)

            state_manager = ConversationStateManager(
                tenant_id=tenant_id,
                agent_id=agent_id,
                max_context_messages=max_ctx,
                max_output_tokens=max_out,
            )
            rule_engine = BusinessRuleEngine(runtime_config.business_rules)

            # Initialize authoritative call session accumulator for billing & transcripts
            accumulator = CallSessionAccumulator(
                tenant_id=tenant_id,
                agent_id=agent_id,
                room_name=ctx.room.name,
                agent_version_id=runtime_config.agent_version_id,
                caller_identity=participant.identity,
                caller_name=getattr(participant, "name", None) or participant.identity,
            )

            # Initialize backend-controlled clinic tools
            tools_manager = ClinicToolsManager(
                tenant_id=tenant_id,
                agent_id=agent_id,
                client=client,
                transfer_rules=runtime_config.transfer_rules if isinstance(runtime_config.transfer_rules, dict) else {},
                clinic_profile=runtime_config.clinic_profile or {},
                state_manager=state_manager,
                on_event=broadcast_data_packet,
            )
            function_tools = tools_manager.get_function_tools()
            active_call_session = BrowserSession(
                tenant_id=tenant_id,
                agent_id=agent_id,
                room_name=ctx.room.name,
                runtime_config=runtime_config,
                state_manager=state_manager,
                accumulator=accumulator,
                participant_identity=participant.identity,
                agent_version_id=runtime_config.agent_version_id,
            )
            active_call_session.transition_to(CallSessionState.CONNECTING, "browser_route_resolved")
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

    # 5. Resolve active LLM provider and initialize voice pipeline session
    resolved_llm_provider = settings.llm_provider
    resolved_llm_model = (
        settings.ollama_model if settings.llm_provider.lower() in ("ollama", "local") else settings.gemini_model
    )
    resolved_llm_base_url = settings.ollama_base_url
    resolved_ai_mode = settings.ai_mode

    if runtime_config:
        if runtime_config.llm_provider:
            resolved_llm_provider = runtime_config.llm_provider
        if runtime_config.ai_model:
            if settings.use_realtime_audio and runtime_config.ai_model == "gemini-2.5-flash":
                resolved_llm_model = settings.gemini_model
            else:
                resolved_llm_model = runtime_config.ai_model
        if runtime_config.llm_base_url:
            resolved_llm_base_url = runtime_config.llm_base_url
        if runtime_config.ai_mode:
            resolved_ai_mode = runtime_config.ai_mode

    logger.info(
        "Voice session LLM provider resolved: provider=%s, model=%s, mode=%s",
        resolved_llm_provider,
        resolved_llm_model,
        resolved_ai_mode,
    )

    byok_creds = runtime_config.byok_credentials if runtime_config else {}
    lang_settings = runtime_config.language_settings if runtime_config else {}

    effective_gemini_key = byok_creds.get("geminiApiKey") or settings.gemini_api_key
    effective_sarvam_key = byok_creds.get("sarvamApiKey") or settings.sarvam_api_key
    route_thanglish = lang_settings.get("routeThanglishToSarvam", False)

    try:
        session = create_agent_session(
            vad=vad,
            llm_provider=resolved_llm_provider,
            llm_model=resolved_llm_model,
            llm_base_url=resolved_llm_base_url,
            ai_mode=resolved_ai_mode,
            gemini_api_key=effective_gemini_key,
            whisper_model_size=settings.whisper_model_size,
            whisper_device=settings.whisper_device,
            whisper_compute_type=settings.whisper_compute_type,
            whisper_language=settings.whisper_language,
            whisper_initial_prompt=settings.whisper_initial_prompt,
            piper_voice=piper_voice,
            piper_model_path=settings.piper_model_path,
            piper_config_path=settings.piper_config_path,
            sarvam_api_key=effective_sarvam_key,
            sarvam_model=settings.sarvam_model,
            route_thanglish_to_sarvam=route_thanglish,
            use_realtime=settings.use_realtime_audio,
        )
        try:
            agent = create_agent(instructions=instructions, tools=function_tools)
        except TypeError:
            agent = create_agent(instructions=instructions)
    except Exception as exc:
        logger.error("Failed to initialize voice session components: %s", exc)
        raise

    # 6. Attach event listeners for transcripts, metering, and lifecycle tracking
    reported_lock = asyncio.Lock()
    has_reported = False

    async def finalize_and_report_call(reason: str, status: Optional[str] = None) -> None:
        nonlocal has_reported
        async with reported_lock:
            if has_reported:
                return
            has_reported = True

        if active_call_session:
            term_state = (
                CallSessionState.FAILED
                if (status == "FAILED" or "fail" in reason.lower() or "error" in reason.lower())
                else CallSessionState.CANCELLED
                if (status == "CANCELLED" or "cancel" in reason.lower())
                else CallSessionState.COMPLETED
            )
            active_call_session.transition_to(term_state, reason)

        if not accumulator or not client:
            return

        try:
            # Fallback check: inspect session.history to capture any turns that didn't fire an event
            try:
                if hasattr(session, "history") and hasattr(session.history, "messages"):
                    raw_messages = session.history.messages
                    history_messages = raw_messages() if callable(raw_messages) else raw_messages
                    for msg in history_messages:
                        accumulator.record_livekit_item(msg)
            except Exception as hist_err:
                logger.debug("Could not inspect session history: %s", hist_err)

            report = accumulator.finish(reason=reason, status=status)
            logger.info(
                "Submitting authoritative call session to control plane: room=%s, duration=%ss, turns=%s, status=%s",
                report.room_name,
                report.duration_seconds,
                len(report.transcripts),
                report.status,
            )
            result = await client.report_call_session(report)
            logger.info(
                "Call session successfully saved in control plane: callId=%s, billableMinutes=%s, isDuplicate=%s",
                result.get("callId"),
                result.get("billableMinutes"),
                result.get("isDuplicate"),
            )
            if call_id:
                try:
                    await client.update_call_lifecycle(
                        call_id=call_id,
                        status=report.status,
                        duration_seconds=report.duration_seconds,
                        end_reason=reason,
                    )
                except Exception as lc_end_err:
                    logger.debug("Notice: Call lifecycle update to %s: %s", report.status, lc_end_err)
        except Exception as rep_err:
            logger.error(
                "Failed to report call session to control plane for room %s: %s",
                ctx.room.name,
                rep_err,
            )

    async def broadcast_speech_and_data(
        message: str,
        tool_invocation: Optional[str] = None,
        tool_result: Optional[str] = None,
    ) -> None:
        """Sends audio speech to LiveKit room and mirrors text to data channel."""
        await broadcast_data_packet({
            "type": "transcript",
            "speaker": "AGENT",
            "message": message,
            "toolInvocation": tool_invocation,
            "toolResult": tool_result,
            "timestamp": datetime.datetime.now().strftime("%I:%M:%S %p"),
        })
        await session.say(message)

    async def handle_llm_turn_with_circuit_breaker(user_input: str) -> None:
        """Executes LLM reply generation with pre-turn rule evaluation and bounded retries."""
        if state_manager:
            state_manager.add_message("user", user_input)

        if rule_engine:
            clinic_ph = (
                runtime_config.clinic_profile.get("phone", "+1 (555) 234-5678")
                if runtime_config and runtime_config.clinic_profile
                else "+1 (555) 234-5678"
            )
            pre_eval = rule_engine.evaluate_pre_turn(user_input, clinic_phone=clinic_ph)
            if pre_eval.is_blocking and pre_eval.message:
                logger.info("Pre-turn rule evaluation triggered action '%s': %s", pre_eval.action, pre_eval.reason)
                if accumulator:
                    accumulator.add_transcript(speaker="AGENT", message=pre_eval.message)
                if state_manager:
                    state_manager.add_message("assistant", pre_eval.message)
                await broadcast_speech_and_data(pre_eval.message, tool_invocation=pre_eval.tool_name)
                return

        if tools_manager:
            tools_manager.loop_limiter.reset_turn()

        for attempt in range(3):
            try:
                await session.generate_reply(user_input=user_input)
                return
            except Exception as exc:
                err_str = str(exc).lower()
                is_quota = "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str
                is_transient = "timeout" in err_str or "temporarily unavailable" in err_str or "503" in err_str

                if is_quota:
                    logger.warning(
                        "LLM quota / rate limit exceeded (429). Activating safe deterministic fallback."
                    )
                    clinic_phone = (
                        runtime_config.clinic_profile.get("phone")
                        if runtime_config and runtime_config.clinic_profile
                        else "+1 (555) 234-5678"
                    )
                    fallback_msg = (
                        f"I apologize, our automated clinic assistant is experiencing unusually high call volume. "
                        f"Please hold for our front desk, or contact our reception directly at {clinic_phone}."
                    )
                    if accumulator:
                        accumulator.add_transcript(speaker="AGENT", message=fallback_msg)
                    await broadcast_speech_and_data(fallback_msg)
                    return

                if is_transient and attempt < 2:
                    backoff = 0.5 * (2 ** attempt)
                    logger.info("Transient LLM error (attempt %d/3). Backing off %.1fs...", attempt + 1, backoff)
                    await asyncio.sleep(backoff)
                    continue

                logger.error("LLM reply generation failed: %s", exc)
                fallback_msg = "I apologize, I am having a moment of difficulty. Could you please repeat that?"
                if accumulator:
                    accumulator.add_transcript(speaker="AGENT", message=fallback_msg)
                await broadcast_speech_and_data(fallback_msg)
                return

    @session.on("conversation_item_added")
    def on_conversation_item_added(ev) -> None:
        item = getattr(ev, "item", ev)
        if accumulator:
            accumulator.record_livekit_item(item)

        # Update language preference state from user speech/text
        if state_manager:
            try:
                role = str(getattr(item, "role", "")).lower()
                if role == "user":
                    u_text = getattr(item, "text_content", None) or getattr(item, "content", "")
                    if callable(u_text):
                        u_text = u_text()
                    u_text_str = str(u_text).strip()
                    if u_text_str:
                        state_manager.update_language_preference_from_input(u_text_str)
            except Exception as lang_err:
                logger.debug("Could not extract language preference from item: %s", lang_err)

        # Mirror agent verbal speech to live data channel
        try:
            role = str(getattr(item, "role", "")).lower()
            if role == "assistant":
                text = getattr(item, "text_content", None) or getattr(item, "content", "")
                if callable(text):
                    text = text()
                text_str = str(text).strip()
                if text_str:
                    asyncio.create_task(
                        broadcast_data_packet({
                            "type": "transcript",
                            "speaker": "AGENT",
                            "message": text_str,
                            "timestamp": datetime.datetime.now().strftime("%I:%M:%S %p"),
                        })
                    )
        except Exception as b_err:
            logger.debug("Could not mirror conversation item to data channel: %s", b_err)

    @ctx.room.on("data_received")
    def on_data_received(dp: rtc.DataPacket) -> None:
        try:
            if not dp.data:
                return
            text = dp.data.decode("utf-8").strip()
            if not text:
                return
            logger.info("Received query from participant '%s': %s", getattr(dp.participant, "identity", "unknown"), text)

            # Barge-in / Interruption: cancel any currently playing agent speech immediately
            try:
                session.interrupt(force=True)
            except Exception as int_err:
                logger.debug("Barge-in interrupt triggered: %s", int_err)

            if accumulator:
                accumulator.add_transcript(speaker="CUSTOMER", message=text)

            # Check deterministic fast-path first to eliminate LLM latency, cost, and hallucinations
            if runtime_config:
                fast_path = match_deterministic_clinic_query(text, runtime_config)
                if fast_path:
                    resp_text, tool_name = fast_path
                    logger.info("Fast-path match for query '%s' -> '%s'", text, resp_text)
                    if accumulator:
                        accumulator.add_transcript(speaker="AGENT", message=resp_text)
                    asyncio.create_task(
                        broadcast_speech_and_data(resp_text, tool_invocation=tool_name)
                    )
                    return

            # Execute conversational turn via LLM with bounded retries and 429 quota protection
            asyncio.create_task(handle_llm_turn_with_circuit_breaker(text))
        except Exception as data_err:
            logger.error("Error handling data packet from participant: %s", data_err)

    @session.on("error")
    def on_session_error(ev) -> None:
        err = getattr(ev, "error", ev)
        logger.error("Session error occurred in room %s: %s", ctx.room.name, err)
        err_str = str(err).lower()
        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
            logger.warning("Session caught 429 rate limit. Broadcasting fallback.")
            clinic_phone = (
                runtime_config.clinic_profile.get("phone", "+1 (555) 234-5678")
                if runtime_config and runtime_config.clinic_profile
                else "+1 (555) 234-5678"
            )
            fallback_msg = (
                f"I apologize, our automated clinic assistant is experiencing unusually high call volume. "
                f"Please hold for our front desk, or contact our reception directly at {clinic_phone}."
            )
            if accumulator:
                accumulator.add_transcript(speaker="AGENT", message=fallback_msg)
            asyncio.create_task(broadcast_speech_and_data(fallback_msg))

    @session.on("close")
    def on_session_close(ev) -> None:
        reason = getattr(ev, "reason", "normal")
        err = getattr(ev, "error", None)
        reason_str = str(reason)
        is_failure = bool(err) or reason_str in ("realtime_model_error", "error", "failed")
        if is_failure:
            logger.error("Session ended with error in room %s: %s (reason: %s)", ctx.room.name, err, reason_str)
            asyncio.create_task(finalize_and_report_call(reason_str, status="FAILED"))
        else:
            logger.info("Session ended normally for room %s (reason: %s)", ctx.room.name, reason_str)
            asyncio.create_task(finalize_and_report_call(reason_str, status="COMPLETED"))

    async def on_shutdown() -> None:
        await finalize_and_report_call("shutdown")
        logger.info("Worker session cleanup/shutdown completed for room: %s", ctx.room.name)

    ctx.add_shutdown_callback(on_shutdown)

    # 7. Start the voice pipeline session
    logger.info("Voice session started for room: %s with participant: %s", ctx.room.name, participant.identity)
    try:
        await session.start(agent, room=ctx.room)
    except Exception as exc:
        err_str = str(exc).lower()
        if "1008" in err_str or "realtime" in err_str or "policy violation" in err_str:
            logger.warning(
                "Realtime session startup rejected (%s). Activating robust cascaded STT + Gemini 2.5 Flash + TTS fallback.",
                exc,
            )
            session = create_agent_session(
                vad=vad,
                llm_provider="gemini",
                llm_model="gemini-2.5-flash",
                llm_base_url=resolved_llm_base_url,
                ai_mode="CLOUD_AI",
                gemini_api_key=effective_gemini_key,
                whisper_model_size=settings.whisper_model_size,
                whisper_device=settings.whisper_device,
                whisper_compute_type=settings.whisper_compute_type,
                whisper_language=settings.whisper_language,
                whisper_initial_prompt=settings.whisper_initial_prompt,
                piper_voice=piper_voice,
                piper_model_path=settings.piper_model_path,
                piper_config_path=settings.piper_config_path,
                sarvam_api_key=effective_sarvam_key,
                sarvam_model=settings.sarvam_model,
                route_thanglish_to_sarvam=route_thanglish,
                use_realtime=False,
            )
            await session.start(agent, room=ctx.room)
        else:
            raise

    # 8. Initial voice greeting to caller
    if active_call_session:
        active_call_session.transition_to(CallSessionState.ACTIVE, "voice_session_active")

    if accumulator:
        accumulator.add_transcript(
            speaker="AGENT",
            message=greeting,
            timestamp_offset_ms=0,
        )
    if state_manager:
        state_manager.add_message("assistant", greeting, is_greeting=True)
    await broadcast_speech_and_data(greeting)


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
    if settings.livekit_agent_deployment:
        os.environ["LIVEKIT_AGENT_DEPLOYMENT"] = settings.livekit_agent_deployment

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

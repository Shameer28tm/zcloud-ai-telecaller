"""Client and models for fetching dynamic agent configuration from the Control Plane runtime API."""

import asyncio
import datetime
import json
import logging
import time
from typing import Any, Optional

import aiohttp
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("zcloud_ai_telecaller.runtime_client")


class RuntimeClientError(Exception):
    """Base exception for all runtime client errors."""


class RuntimeClientAuthError(RuntimeClientError):
    """Raised when authentication to the control plane fails (401)."""


class RuntimeClientNotFoundError(RuntimeClientError):
    """Raised when tenant, agent, or version is not found (404)."""


class RuntimeClientValidationError(RuntimeClientError):
    """Raised when runtime request is invalid or version is unpublished (400)."""


class RuntimeClientConnectionError(RuntimeClientError):
    """Raised on network failures, timeouts, or server errors (5xx)."""


class AgentRuntimeConfig(BaseModel):
    """Runtime configuration for an active agent version fetched dynamically from the control plane."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    tenant_id: str = Field(validation_alias="tenantId")
    agent_id: str = Field(validation_alias="agentId")
    agent_version_id: str = Field(validation_alias="agentVersionId")
    name: str
    prompt: str
    greeting: str
    closing_message: Optional[str] = Field(default=None, validation_alias="closingMessage")
    personality: Optional[str] = None
    tone: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = Field(default=500, validation_alias="maxTokens")
    voice_provider: str = Field(default="piper", validation_alias="voiceProvider")
    voice_id: str = Field(default="en_US-lessac-medium", validation_alias="voiceId")
    voice_language: str = Field(default="en-US", validation_alias="voiceLanguage")
    ai_model: str = Field(default="gemini-2.5-flash", validation_alias="aiModel")
    llm_provider: Optional[str] = Field(default=None, validation_alias="llmProvider")
    llm_base_url: Optional[str] = Field(default=None, validation_alias="llmBaseUrl")
    ai_mode: Optional[str] = Field(default=None, validation_alias="aiMode")
    stt_provider: str = Field(default="faster-whisper", validation_alias="sttProvider")
    tools_config: Any = Field(default_factory=dict, validation_alias="toolsConfig")
    business_rules: Optional[dict[str, Any]] = Field(default=None, validation_alias="businessRules")
    business_hours: Optional[Any] = Field(default=None, validation_alias="businessHours")
    transfer_rules: Any = Field(default_factory=dict, validation_alias="transferRules")
    clinic_profile: Optional[dict[str, Any]] = Field(default=None, validation_alias="clinicProfile")
    knowledge_base: list[dict[str, Any]] = Field(default_factory=list, validation_alias="knowledgeBase")


class CallTranscriptItem(BaseModel):
    """Individual transcript turn recorded during a call session."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    speaker: str = "AGENT"  # "AGENT" | "CUSTOMER" | "SYSTEM"
    message: str
    timestamp_offset_ms: int = Field(default=0, validation_alias="timestampOffsetMs", serialization_alias="timestampOffsetMs")
    latency_ms: Optional[int] = Field(default=None, validation_alias="latencyMs", serialization_alias="latencyMs")
    confidence: Optional[float] = None


class CallSessionReport(BaseModel):
    """Complete call session payload submitted authoritatively by worker upon call termination."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    tenant_id: str = Field(validation_alias="tenantId", serialization_alias="tenantId")
    agent_id: str = Field(validation_alias="agentId", serialization_alias="agentId")
    agent_version_id: Optional[str] = Field(default=None, validation_alias="agentVersionId", serialization_alias="agentVersionId")
    room_name: str = Field(validation_alias="roomName", serialization_alias="roomName")
    caller_identity: Optional[str] = Field(default=None, validation_alias="callerIdentity", serialization_alias="callerIdentity")
    caller_number: Optional[str] = Field(default=None, validation_alias="callerNumber", serialization_alias="callerNumber")
    recipient_number: Optional[str] = Field(default=None, validation_alias="recipientNumber", serialization_alias="recipientNumber")
    caller_name: Optional[str] = Field(default=None, validation_alias="callerName", serialization_alias="callerName")
    direction: str = "INBOUND"
    started_at: Optional[str] = Field(default=None, validation_alias="startedAt", serialization_alias="startedAt")
    ended_at: Optional[str] = Field(default=None, validation_alias="endedAt", serialization_alias="endedAt")
    duration_seconds: int = Field(default=0, validation_alias="durationSeconds", serialization_alias="durationSeconds")
    status: str = "COMPLETED"
    intent: Optional[str] = None
    outcome: Optional[str] = None
    summary: Optional[str] = None
    sentiment: Optional[str] = None
    end_reason: Optional[str] = Field(default=None, validation_alias="endReason", serialization_alias="endReason")
    transcripts: list[CallTranscriptItem] = Field(default_factory=list)


class CallSessionAccumulator:
    """Accumulates transcript turns, offsets, and call duration for authoritative session reporting."""

    def __init__(
        self,
        tenant_id: str,
        agent_id: str,
        room_name: str,
        agent_version_id: Optional[str] = None,
        caller_identity: Optional[str] = None,
        caller_number: Optional[str] = None,
        caller_name: Optional[str] = None,
        start_time: Optional[float] = None,
        recipient_number: Optional[str] = None,
        direction: str = "INBOUND",
    ) -> None:
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.room_name = room_name
        self.agent_version_id = agent_version_id
        self.caller_identity = caller_identity
        self.caller_number = caller_number
        self.caller_name = caller_name
        self.recipient_number = recipient_number
        self.direction = direction
        self.start_time = start_time if start_time is not None else time.time()
        self.end_time: Optional[float] = None
        self.end_reason: Optional[str] = None
        self.status: str = "COMPLETED"
        self.transcripts: list[CallTranscriptItem] = []
        self._seen_messages: dict[tuple[str, str], int] = {}

    def add_transcript(
        self,
        speaker: str,
        message: str,
        timestamp_offset_ms: Optional[int] = None,
        latency_ms: Optional[int] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        """Appends a transcript turn with offset, discarding empty or duplicate turns."""
        clean_msg = message.strip()
        if not clean_msg:
            return False

        if timestamp_offset_ms is None:
            now = time.time()
            timestamp_offset_ms = max(0, int((now - self.start_time) * 1000))

        # Deduplicate identical message from the same speaker within 4 seconds
        dedup_key = (speaker, clean_msg)
        if dedup_key in self._seen_messages:
            last_ts = self._seen_messages[dedup_key]
            if abs(timestamp_offset_ms - last_ts) < 4000:
                return False

        self._seen_messages[dedup_key] = timestamp_offset_ms
        self.transcripts.append(
            CallTranscriptItem(
                speaker=speaker,
                message=clean_msg,
                timestamp_offset_ms=timestamp_offset_ms,
                latency_ms=latency_ms,
                confidence=confidence,
            )
        )
        return True

    def record_livekit_item(
        self,
        item: Any,
        timestamp_offset_ms: Optional[int] = None,
    ) -> bool:
        """Extracts speaker and text from a LiveKit chat item or event and records it."""
        # If item is an event wrapping an inner message (like ConversationItemAddedEvent), unwrap it
        if not hasattr(item, "role") and hasattr(item, "item"):
            inner = getattr(item, "item", None)
            if inner is not None:
                item = inner
        elif hasattr(item, "item") and hasattr(item, "created_at") and not isinstance(getattr(item, "role", None), str):
            # LiveKit ConversationItemAddedEvent has .item and .created_at
            inner = getattr(item, "item", None)
            if inner is not None:
                item = inner

        raw_role = getattr(item, "role", None)
        role = str(raw_role or "").lower()
        if role in ("system", "developer"):
            return False

        speaker = "CUSTOMER" if role == "user" else "AGENT"

        text = ""
        if hasattr(item, "text_content"):
            text_val = item.text_content
            if callable(text_val) and not isinstance(text_val, (str, bytes)):
                text = str(text_val())
            else:
                text = str(text_val)
        elif hasattr(item, "content"):
            content = getattr(item, "content")
            if isinstance(content, list):
                text = " ".join(str(c) for c in content if c)
            else:
                text = str(content)

        text = text.strip()
        if not text:
            return False

        confidence_val = getattr(item, "transcript_confidence", None)
        confidence: Optional[float] = None
        if isinstance(confidence_val, (int, float)):
            confidence = float(confidence_val)

        latency_val = getattr(item, "latency_ms", None)
        latency_ms: Optional[int] = None
        if isinstance(latency_val, (int, float)):
            latency_ms = int(latency_val)

        return self.add_transcript(
            speaker=speaker,
            message=text,
            timestamp_offset_ms=timestamp_offset_ms,
            latency_ms=latency_ms,
            confidence=confidence,
        )

    def finish(
        self,
        reason: Optional[str] = None,
        status: str = "COMPLETED",
        end_time: Optional[float] = None,
    ) -> CallSessionReport:
        """Finalizes timestamps and builds the authoritative CallSessionReport."""
        self.end_time = end_time if end_time is not None else time.time()
        self.end_reason = reason or "normal"
        self.status = status

        duration_seconds = max(0, int(round(self.end_time - self.start_time)))

        started_at = datetime.datetime.fromtimestamp(
            self.start_time, tz=datetime.timezone.utc
        ).isoformat()
        ended_at = datetime.datetime.fromtimestamp(
            self.end_time, tz=datetime.timezone.utc
        ).isoformat()

        return CallSessionReport(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            agent_version_id=self.agent_version_id,
            room_name=self.room_name,
            caller_identity=self.caller_identity,
            caller_number=self.caller_number,
            caller_name=self.caller_name,
            recipient_number=self.recipient_number,
            direction=self.direction,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            status=self.status,
            end_reason=self.end_reason,
            transcripts=list(self.transcripts),
        )


def extract_participant_metadata(metadata_raw: Optional[str]) -> tuple[str, str, Optional[str]]:
    """Extracts and validates (tenant_id, agent_id, agent_version_id) from raw participant metadata.

    Args:
        metadata_raw: Raw JSON string from participant.metadata.

    Returns:
        tuple[str, str, Optional[str]]: (tenant_id, agent_id, agent_version_id)

    Raises:
        RuntimeClientValidationError: If metadata is empty, malformed JSON, or missing tenantId / agentId.
    """
    if not metadata_raw or not metadata_raw.strip():
        raise RuntimeClientValidationError("Participant metadata is empty or missing")

    try:
        data = json.loads(metadata_raw)
    except Exception as exc:
        raise RuntimeClientValidationError(f"Failed to parse participant metadata JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeClientValidationError("Participant metadata must be a JSON object")

    tenant_id = data.get("tenantId") or data.get("tenant_id")
    agent_id = data.get("agentId") or data.get("agent_id")
    agent_version_id = data.get("agentVersionId") or data.get("agent_version_id")

    if not tenant_id or not str(tenant_id).strip():
        raise RuntimeClientValidationError("Participant metadata missing required 'tenantId'")
    if not agent_id or not str(agent_id).strip():
        raise RuntimeClientValidationError("Participant metadata missing required 'agentId'")

    version_str = str(agent_version_id).strip() if agent_version_id else None
    return str(tenant_id).strip(), str(agent_id).strip(), version_str


def extract_sip_participant_info(participant: Any) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extracts (called_number, caller_number, call_id) from a LiveKit SIP participant.

    Inspects participant.attributes (standard LiveKit SIP attributes) and participant identity.

    Returns:
        tuple[Optional[str], Optional[str], Optional[str]]: (called_number, caller_number, call_id)
    """
    called_number = None
    caller_number = None
    call_id = None

    attributes = getattr(participant, "attributes", None)
    if isinstance(attributes, dict):
        called_number = attributes.get("sip.trunkPhoneNumber") or attributes.get("sip.calledNumber")
        caller_number = attributes.get("sip.phoneNumber") or attributes.get("sip.callerNumber")
        call_id = attributes.get("sip.callID") or attributes.get("sip.callId")

    identity = str(getattr(participant, "identity", "") or "")
    if not caller_number and identity.startswith("sip_"):
        caller_number = identity[4:]

    return called_number, caller_number, call_id


def is_sip_session(participant: Any, room: Optional[Any] = None) -> bool:
    """Checks whether the session is an inbound telephony SIP call.

    Evaluates participant kind, SIP identity prefixes, attributes, and room name conventions.
    """
    kind = getattr(participant, "kind", None)
    if str(kind).upper() in ("SIP", "PARTICIPANT_KIND_SIP", "4"):
        return True

    identity = str(getattr(participant, "identity", "") or "").lower()
    if identity.startswith("sip_"):
        return True

    attributes = getattr(participant, "attributes", None)
    if isinstance(attributes, dict) and any(k.startswith("sip.") for k in attributes.keys()):
        return True

    if room and hasattr(room, "name"):
        room_name = str(room.name).lower()
        if room_name.startswith("sip-") or room_name.startswith("sip_"):
            return True

    return False


class RuntimeConfigClient:
    """Client for fetching dynamic agent configuration from the control plane internal runtime API."""

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        worker_api_secret: Optional[str] = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.worker_api_secret = worker_api_secret
        self.timeout_seconds = timeout_seconds

    async def fetch_agent_runtime(
        self,
        tenant_id: str,
        agent_id: str,
        agent_version_id: Optional[str] = None,
    ) -> AgentRuntimeConfig:
        """Fetches the published agent runtime configuration from the control plane.

        Args:
            tenant_id: Target tenant UUID.
            agent_id: Target agent UUID.
            agent_version_id: Optional specific published version UUID.

        Returns:
            AgentRuntimeConfig with dynamic prompt, greeting, and model parameters.

        Raises:
            RuntimeClientValidationError: If query parameters are invalid or version is unpublished.
            RuntimeClientAuthError: If authentication fails (invalid/missing secret).
            RuntimeClientNotFoundError: If tenant, agent, or version is not found.
            RuntimeClientConnectionError: If network request fails or server errors out.
        """
        if not tenant_id or not tenant_id.strip():
            raise RuntimeClientValidationError("tenant_id must be non-empty")
        if not agent_id or not agent_id.strip():
            raise RuntimeClientValidationError("agent_id must be non-empty")
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/agent-runtime"
        params = {
            "tenantId": tenant_id.strip(),
            "agentId": agent_id.strip(),
        }
        if agent_version_id and agent_version_id.strip():
            params["agentVersionId"] = agent_version_id.strip()

        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Accept": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        try:
                            body = await resp.json()
                        except Exception as json_err:
                            raise RuntimeClientConnectionError(
                                f"Invalid JSON received from control plane: {json_err}"
                            ) from json_err

                        if not body.get("success") or "data" not in body:
                            msg = body.get("error", {}).get("message", "Malformed response from control plane")
                            raise RuntimeClientValidationError(msg)

                        return AgentRuntimeConfig.model_validate(body["data"])

                    if resp.status == 401:
                        raise RuntimeClientAuthError("Control plane rejected worker API secret (401 Unauthorized)")

                    if resp.status == 404:
                        raise RuntimeClientNotFoundError(
                            f"Agent or version not found in control plane for tenant={tenant_id}, agent={agent_id} (404 Not Found)"
                        )

                    if resp.status == 400:
                        try:
                            err_body = await resp.json()
                            err_msg = err_body.get("error", {}).get("message", "Bad request to control plane")
                        except Exception:
                            err_msg = "Bad request to control plane (400)"
                        raise RuntimeClientValidationError(err_msg)

                    if resp.status == 403:
                        try:
                            err_body = await resp.json()
                            err_msg = err_body.get("error", {}).get("message", "Tenant is inactive or suspended")
                        except Exception:
                            err_msg = "Tenant is inactive or suspended (403)"
                        raise RuntimeClientValidationError(err_msg)

                    raise RuntimeClientConnectionError(
                        f"Control plane returned unexpected HTTP status {resp.status}"
                    )
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(
                f"Failed to communicate with control plane at {self.base_url}: {net_err}"
            ) from net_err

    async def report_call_session(
        self,
        report: CallSessionReport,
    ) -> dict[str, Any]:
        """Reports completed call session details, transcripts, and duration to the control plane.

        Args:
            report: CallSessionReport containing call metadata, duration, and transcripts.

        Returns:
            dict containing response data (e.g. callId, billableMinutes, transcriptCount).

        Raises:
            RuntimeClientValidationError: If payload is invalid (400) or tenant is inactive (403).
            RuntimeClientAuthError: If authentication fails (401).
            RuntimeClientNotFoundError: If tenant or agent is not found (404).
            RuntimeClientConnectionError: If network request fails or server errors out.
        """
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/call-session"
        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = report.model_dump(by_alias=True, exclude_none=True)
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        try:
                            body = await resp.json()
                        except Exception as json_err:
                            raise RuntimeClientConnectionError(
                                f"Invalid JSON received from control plane: {json_err}"
                            ) from json_err

                        if not body.get("success") or "data" not in body:
                            msg = body.get("error", {}).get("message", "Malformed response from control plane")
                            raise RuntimeClientValidationError(msg)

                        return body["data"]

                    if resp.status == 401:
                        raise RuntimeClientAuthError("Control plane rejected worker API secret (401 Unauthorized)")

                    if resp.status == 404:
                        raise RuntimeClientNotFoundError(
                            f"Tenant or agent not found in control plane for call session {report.room_name} (404 Not Found)"
                        )

                    if resp.status == 400:
                        try:
                            err_body = await resp.json()
                            err_msg = err_body.get("error", {}).get("message", "Bad call-session request to control plane")
                        except Exception:
                            err_msg = "Bad call-session request to control plane (400)"
                        raise RuntimeClientValidationError(err_msg)

                    if resp.status == 403:
                        try:
                            err_body = await resp.json()
                            err_msg = err_body.get("error", {}).get("message", "Tenant is inactive or suspended")
                        except Exception:
                            err_msg = "Tenant is inactive or suspended (403)"
                        raise RuntimeClientValidationError(err_msg)

                    raise RuntimeClientConnectionError(
                        f"Control plane returned unexpected HTTP status {resp.status}"
                    )
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(
                f"Failed to report call session to control plane at {self.base_url}: {net_err}"
            ) from net_err

    async def check_availability(
        self,
        tenant_id: str,
        date: str,
        service_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Queries authoritative clinic appointment availability from control plane."""
        if not tenant_id or not tenant_id.strip():
            raise RuntimeClientValidationError("tenant_id must be non-empty")
        if not date or not date.strip():
            raise RuntimeClientValidationError("date must be non-empty")
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/clinic/availability"
        payload: dict[str, Any] = {
            "tenantId": tenant_id.strip(),
            "date": date.strip(),
        }
        if service_name and service_name.strip():
            payload["serviceName"] = service_name.strip()

        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        if not body.get("success") or "data" not in body:
                            msg = body.get("error", {}).get("message", "Malformed response from availability API")
                            raise RuntimeClientValidationError(msg)
                        return body["data"]

                    if resp.status == 401:
                        raise RuntimeClientAuthError("Control plane rejected worker API secret (401 Unauthorized)")
                    if resp.status == 404:
                        raise RuntimeClientNotFoundError(f"Tenant {tenant_id} not found (404)")
                    if resp.status in (400, 403):
                        body = await resp.json()
                        raise RuntimeClientValidationError(body.get("error", {}).get("message", "Invalid availability request"))

                    raise RuntimeClientConnectionError(f"Availability API returned status {resp.status}")
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(f"Failed to reach availability API at {self.base_url}: {net_err}") from net_err

    async def book_appointment(
        self,
        tenant_id: str,
        patient_name: str,
        phone: str,
        date: str,
        time: str,
        service_name: str = "General Consultation",
        notes: Optional[str] = None,
        agent_id: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Books an appointment in the control plane database with double-booking prevention."""
        if not tenant_id or not tenant_id.strip():
            raise RuntimeClientValidationError("tenant_id must be non-empty")
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/clinic/book"
        payload: dict[str, Any] = {
            "tenantId": tenant_id.strip(),
            "patientName": patient_name.strip(),
            "phone": phone.strip(),
            "date": date.strip(),
            "time": time.strip(),
            "serviceName": service_name.strip(),
        }
        if notes:
            payload["notes"] = notes.strip()
        if agent_id:
            payload["agentId"] = agent_id.strip()
        if call_id:
            payload["callId"] = call_id.strip()

        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        if not body.get("success") or "data" not in body:
                            msg = body.get("error", {}).get("message", "Malformed response from booking API")
                            raise RuntimeClientValidationError(msg)
                        return body["data"]

                    if resp.status == 409:
                        body = await resp.json()
                        err_msg = body.get("error", {}).get("message", "Requested time slot is already booked")
                        raise RuntimeClientValidationError(err_msg)

                    if resp.status == 401:
                        raise RuntimeClientAuthError("Control plane rejected worker API secret (401 Unauthorized)")
                    if resp.status == 404:
                        raise RuntimeClientNotFoundError(f"Tenant {tenant_id} not found (404)")
                    if resp.status in (400, 403):
                        body = await resp.json()
                        err_msg = body.get("error", {}).get("message", "Bad request to booking API")
                        raise RuntimeClientValidationError(err_msg)

                    raise RuntimeClientConnectionError(f"Booking API returned status {resp.status}")
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(f"Failed to reach booking API at {self.base_url}: {net_err}") from net_err

    async def cancel_appointment(
        self,
        tenant_id: str,
        appointment_id: str,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        """Cancels an appointment in the control plane database and frees the slot."""
        if not tenant_id or not tenant_id.strip():
            raise RuntimeClientValidationError("tenant_id must be non-empty")
        if not appointment_id or not appointment_id.strip():
            raise RuntimeClientValidationError("appointment_id must be non-empty")
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/clinic/cancel"
        payload = {
            "tenantId": tenant_id.strip(),
            "appointmentId": appointment_id.strip(),
            "reason": (reason or "Customer requested cancellation").strip(),
        }
        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        return body.get("data", {})
                    if resp.status == 401:
                        raise RuntimeClientAuthError("Unauthorized cancel request")
                    body = await resp.json()
                    raise RuntimeClientValidationError(body.get("error", {}).get("message", "Cancel failed"))
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(f"Failed to reach cancellation API: {net_err}") from net_err

    async def reschedule_appointment(
        self,
        tenant_id: str,
        appointment_id: str,
        new_date: str,
        new_time: str,
    ) -> dict[str, Any]:
        """Reschedules an appointment to a new open slot in the control plane database."""
        if not tenant_id or not tenant_id.strip():
            raise RuntimeClientValidationError("tenant_id must be non-empty")
        if not appointment_id or not appointment_id.strip():
            raise RuntimeClientValidationError("appointment_id must be non-empty")
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/clinic/reschedule"
        payload = {
            "tenantId": tenant_id.strip(),
            "appointmentId": appointment_id.strip(),
            "newDate": new_date.strip(),
            "newTime": new_time.strip(),
        }
        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        return body.get("data", {})
                    if resp.status == 409:
                        body = await resp.json()
                        raise RuntimeClientValidationError(body.get("error", {}).get("message", "Slot unavailable"))
                    if resp.status == 401:
                        raise RuntimeClientAuthError("Unauthorized reschedule request")
                    body = await resp.json()
                    raise RuntimeClientValidationError(body.get("error", {}).get("message", "Reschedule failed"))
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(f"Failed to reach rescheduling API: {net_err}") from net_err

    async def get_clinic_info(self, tenant_id: str) -> dict[str, Any]:
        """Retrieves authoritative clinic profile, address, and contact details."""
        if not tenant_id or not tenant_id.strip():
            raise RuntimeClientValidationError("tenant_id must be non-empty")
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/clinic/info"
        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"tenantId": tenant_id.strip()}

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        return body.get("data", {})
                    if resp.status == 401:
                        raise RuntimeClientAuthError("Unauthorized clinic info request")
                    if resp.status == 404:
                        raise RuntimeClientNotFoundError(f"Tenant {tenant_id} not found")
                    body = await resp.json()
                    raise RuntimeClientValidationError(body.get("error", {}).get("message", "Failed to fetch clinic info"))
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(f"Failed to reach clinic info API: {net_err}") from net_err

    async def get_services(self, tenant_id: str) -> list[dict[str, Any]]:
        """Retrieves active clinic services and durations for the tenant."""
        if not tenant_id or not tenant_id.strip():
            raise RuntimeClientValidationError("tenant_id must be non-empty")
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/clinic/services"
        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"tenantId": tenant_id.strip()}

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        return body.get("data", {}).get("services", [])
                    if resp.status == 401:
                        raise RuntimeClientAuthError("Unauthorized services request")
                    if resp.status == 404:
                        raise RuntimeClientNotFoundError(f"Tenant {tenant_id} not found")
                    body = await resp.json()
                    raise RuntimeClientValidationError(body.get("error", {}).get("message", "Failed to fetch services"))
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(f"Failed to reach services API: {net_err}") from net_err

    async def get_business_hours(self, tenant_id: str) -> dict[str, Any]:
        """Retrieves weekly business hours and schedule for the tenant."""
        if not tenant_id or not tenant_id.strip():
            raise RuntimeClientValidationError("tenant_id must be non-empty")
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/clinic/hours"
        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"tenantId": tenant_id.strip()}

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        return body.get("data", {}).get("schedule", {})
                    if resp.status == 401:
                        raise RuntimeClientAuthError("Unauthorized business hours request")
                    if resp.status == 404:
                        raise RuntimeClientNotFoundError(f"Tenant {tenant_id} not found")
                    body = await resp.json()
                    raise RuntimeClientValidationError(body.get("error", {}).get("message", "Failed to fetch business hours"))
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(f"Failed to reach business hours API: {net_err}") from net_err

    async def find_customer(
        self,
        tenant_id: str,
        phone: Optional[str] = None,
        email: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Queries for existing customer by phone or email in tenant-isolated CRM."""
        if not tenant_id or not tenant_id.strip():
            raise RuntimeClientValidationError("tenant_id must be non-empty")
        if not phone and not email:
            raise RuntimeClientValidationError("Either phone or email must be provided to find customer")
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/clinic/customer/find"
        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: dict[str, Any] = {"tenantId": tenant_id.strip()}
        if phone:
            payload["phone"] = phone.strip()
        if email:
            payload["email"] = email.strip()

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        return body.get("data", {}).get("customer")
                    if resp.status == 401:
                        raise RuntimeClientAuthError("Unauthorized find customer request")
                    body = await resp.json()
                    raise RuntimeClientValidationError(body.get("error", {}).get("message", "Find customer error"))
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(f"Failed to reach find customer API: {net_err}") from net_err

    async def create_customer(
        self,
        tenant_id: str,
        name: str,
        phone: str,
        email: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """Creates or retrieves a customer record in tenant-isolated CRM with duplicate prevention."""
        if not tenant_id or not tenant_id.strip():
            raise RuntimeClientValidationError("tenant_id must be non-empty")
        if not name or not name.strip():
            raise RuntimeClientValidationError("name must be non-empty")
        if not phone or not phone.strip():
            raise RuntimeClientValidationError("phone must be non-empty")
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/clinic/customer/create"
        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: dict[str, Any] = {
            "tenantId": tenant_id.strip(),
            "name": name.strip(),
            "phone": phone.strip(),
        }
        if email:
            payload["email"] = email.strip()
        if notes:
            payload["notes"] = notes.strip()

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        return body.get("data", {})
                    if resp.status == 401:
                        raise RuntimeClientAuthError("Unauthorized create customer request")
                    body = await resp.json()
                    raise RuntimeClientValidationError(body.get("error", {}).get("message", "Create customer error"))
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(f"Failed to reach create customer API: {net_err}") from net_err

    async def request_human_transfer(
        self,
        tenant_id: str,
        call_id: Optional[str] = None,
        reason: Optional[str] = None,
        caller_number: Optional[str] = None,
    ) -> dict[str, Any]:
        """Requests human staff transfer and records escalation CallEvent."""
        if not tenant_id or not tenant_id.strip():
            raise RuntimeClientValidationError("tenant_id must be non-empty")
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/clinic/transfer"
        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: dict[str, Any] = {"tenantId": tenant_id.strip()}
        if call_id:
            payload["callId"] = call_id.strip()
        if reason:
            payload["reason"] = reason.strip()
        if caller_number:
            payload["callerNumber"] = caller_number.strip()

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        return body.get("data", {})
                    if resp.status == 401:
                        raise RuntimeClientAuthError("Unauthorized transfer request")
                    body = await resp.json()
                    raise RuntimeClientValidationError(body.get("error", {}).get("message", "Transfer error"))
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(f"Failed to reach transfer API: {net_err}") from net_err

    async def resolve_sip_inbound(
        self,
        called_number: str,
        caller_number: Optional[str] = None,
        call_id: Optional[str] = None,
        sip_trunk_id: Optional[str] = None,
        room_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Resolves inbound SIP call routing authoritatively from the control plane.

        Security:
            Resolves tenant and agent strictly from the called phone number (DNIS).
            The caller cannot forge or supply tenantId.
        """
        if not called_number or not str(called_number).strip():
            raise RuntimeClientValidationError("called_number must be non-empty for inbound SIP routing")
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/sip/inbound"
        payload = {
            "calledNumber": str(called_number).strip(),
            "callerNumber": str(caller_number).strip() if caller_number else None,
            "callId": str(call_id).strip() if call_id else None,
            "sipTrunkId": str(sip_trunk_id).strip() if sip_trunk_id else None,
            "roomName": str(room_name).strip() if room_name else None,
        }
        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        try:
                            body = await resp.json()
                        except Exception as json_err:
                            raise RuntimeClientConnectionError(
                                f"Invalid JSON received from control plane: {json_err}"
                            ) from json_err

                        if not body.get("success") or "data" not in body:
                            msg = body.get("error", {}).get("message", "Malformed response from control plane")
                            raise RuntimeClientValidationError(msg)

                        data = body["data"]
                        if "runtimeConfig" in data and isinstance(data["runtimeConfig"], dict):
                            data["parsedRuntimeConfig"] = AgentRuntimeConfig.model_validate(data["runtimeConfig"])
                        return data

                    if resp.status == 401:
                        raise RuntimeClientAuthError("Control plane rejected worker API secret (401 Unauthorized)")

                    if resp.status == 404:
                        try:
                            err_body = await resp.json()
                            err_msg = err_body.get("error", {}).get("message", "Phone number or agent not found (404)")
                        except Exception:
                            err_msg = f"Phone number {called_number} or agent not found (404)"
                        raise RuntimeClientNotFoundError(err_msg)

                    if resp.status in (400, 402, 403):
                        try:
                            err_body = await resp.json()
                            err_msg = err_body.get("error", {}).get("message", f"Request rejected ({resp.status})")
                        except Exception:
                            err_msg = f"Request rejected ({resp.status})"
                        raise RuntimeClientValidationError(err_msg)

                    raise RuntimeClientConnectionError(
                        f"Control plane returned unexpected HTTP status {resp.status}"
                    )
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(
                f"Failed to communicate with control plane at {self.base_url}: {net_err}"
            ) from net_err

    async def update_call_lifecycle(
        self,
        call_id: str,
        status: str,
        end_reason: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        failure_reason: Optional[str] = None,
        outcome: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> dict[str, Any]:
        """Updates the call lifecycle state in the control plane."""
        if not call_id or not str(call_id).strip():
            raise RuntimeClientValidationError("call_id must be non-empty")
        if not self.worker_api_secret:
            raise RuntimeClientAuthError("Worker API secret is not configured")

        url = f"{self.base_url}/api/internal/sip/lifecycle"
        payload: dict[str, Any] = {
            "callId": str(call_id).strip(),
            "status": str(status).strip(),
        }
        if end_reason:
            payload["endReason"] = end_reason
        if duration_seconds is not None:
            payload["durationSeconds"] = duration_seconds
        if failure_reason:
            payload["failureReason"] = failure_reason
        if outcome:
            payload["outcome"] = outcome
        if summary:
            payload["summary"] = summary

        headers = {
            "Authorization": f"Bearer {self.worker_api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        return body.get("data", {})
                    if resp.status == 401:
                        raise RuntimeClientAuthError("Control plane rejected worker API secret (401)")
                    if resp.status == 404:
                        raise RuntimeClientNotFoundError(f"Call {call_id} not found")
                    raise RuntimeClientConnectionError(f"Unexpected status {resp.status} from control plane")
        except (RuntimeClientError,):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
            raise RuntimeClientConnectionError(f"Failed to update lifecycle: {net_err}") from net_err


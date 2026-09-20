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
    stt_provider: str = Field(default="faster-whisper", validation_alias="sttProvider")
    tools_config: Any = Field(default_factory=dict, validation_alias="toolsConfig")
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
    ) -> None:
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.room_name = room_name
        self.agent_version_id = agent_version_id
        self.caller_identity = caller_identity
        self.caller_number = caller_number
        self.caller_name = caller_name
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
            direction="INBOUND",
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

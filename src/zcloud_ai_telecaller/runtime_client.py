"""Client and models for fetching dynamic agent configuration from the Control Plane runtime API."""

import asyncio
import json
import logging
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

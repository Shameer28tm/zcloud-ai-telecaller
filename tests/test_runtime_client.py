"""Tests for Phase 2A Runtime Config Client, metadata extraction, and dynamic agent prompt injection."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zcloud_ai_telecaller.runtime_client import (
    AgentRuntimeConfig,
    RuntimeClientAuthError,
    RuntimeClientConnectionError,
    RuntimeClientError,
    RuntimeClientNotFoundError,
    RuntimeClientValidationError,
    RuntimeConfigClient,
    extract_participant_metadata,
)


def test_extract_participant_metadata_camel_case():
    """Verify metadata extraction with standard camelCase keys."""
    raw = json.dumps({
        "tenantId": "tenant-123",
        "agentId": "agent-456",
        "agentVersionId": "version-789",
    })
    t_id, a_id, v_id = extract_participant_metadata(raw)
    assert t_id == "tenant-123"
    assert a_id == "agent-456"
    assert v_id == "version-789"


def test_extract_participant_metadata_snake_case():
    """Verify metadata extraction with snake_case keys."""
    raw = json.dumps({
        "tenant_id": "tenant-abc",
        "agent_id": "agent-def",
        "agent_version_id": "version-ghi",
    })
    t_id, a_id, v_id = extract_participant_metadata(raw)
    assert t_id == "tenant-abc"
    assert a_id == "agent-def"
    assert v_id == "version-ghi"


def test_extract_participant_metadata_optional_version():
    """Verify metadata extraction when version ID is omitted."""
    raw = json.dumps({
        "tenantId": "tenant-123",
        "agentId": "agent-456",
    })
    t_id, a_id, v_id = extract_participant_metadata(raw)
    assert t_id == "tenant-123"
    assert a_id == "agent-456"
    assert v_id is None


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "   ",
        None,
        "not-valid-json",
        "12345",
        "[]",
        json.dumps({"tenantId": "t1"}),  # missing agentId
        json.dumps({"agentId": "a1"}),  # missing tenantId
        json.dumps({"tenantId": "", "agentId": "a1"}),
    ],
)
def test_extract_participant_metadata_invalid(invalid_input):
    """Verify validation error raised for malformed or incomplete metadata."""
    with pytest.raises(RuntimeClientValidationError):
        extract_participant_metadata(invalid_input)


def test_agent_runtime_config_model():
    """Verify AgentRuntimeConfig parses camelCase and snake_case API payloads."""
    payload = {
        "tenantId": "tenant-001",
        "agentId": "agent-002",
        "agentVersionId": "ver-003",
        "name": "Dr. Smith Receptionist",
        "prompt": "You are a dental clinic receptionist.",
        "greeting": "Good morning! How may I assist you with your dental health today?",
        "temperature": 0.5,
        "maxTokens": 300,
        "voiceProvider": "piper",
        "voiceId": "en_US-lessac-medium",
        "aiModel": "gemini-2.5-flash",
        "sttProvider": "faster-whisper",
    }
    config = AgentRuntimeConfig.model_validate(payload)
    assert config.tenant_id == "tenant-001"
    assert config.agent_id == "agent-002"
    assert config.agent_version_id == "ver-003"
    assert config.name == "Dr. Smith Receptionist"
    assert "receptionist" in config.prompt
    assert "Good morning" in config.greeting
    assert config.temperature == 0.5
    assert config.max_tokens == 300
    assert config.voice_id == "en_US-lessac-medium"
    assert config.ai_model == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_fetch_agent_runtime_success():
    """Verify successful runtime config retrieval via mock HTTP session."""
    client = RuntimeConfigClient(
        base_url="http://localhost:3000",
        worker_api_secret="test_secret_123",
    )

    mock_data = {
        "success": True,
        "data": {
            "tenantId": "tenant-real-1",
            "agentId": "agent-real-2",
            "agentVersionId": "version-real-3",
            "name": "Test Clinic Agent",
            "prompt": "Custom prompt dynamically loaded from dashboard.",
            "greeting": "Welcome to our live clinic!",
            "voiceProvider": "piper",
            "voiceId": "en_US-lessac-medium",
            "aiModel": "gemini-2.5-flash",
        },
    }

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=mock_data)

    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_get.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_get)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        config = await client.fetch_agent_runtime(
            tenant_id="tenant-real-1",
            agent_id="agent-real-2",
            agent_version_id="version-real-3",
        )

    assert config.name == "Test Clinic Agent"
    assert config.prompt == "Custom prompt dynamically loaded from dashboard."
    assert config.greeting == "Welcome to our live clinic!"

    # Verify headers and params passed to get()
    call_args = mock_session.get.call_args
    assert call_args is not None
    url, kwargs = call_args[0][0], call_args[1]
    assert url == "http://localhost:3000/api/internal/agent-runtime"
    assert kwargs["headers"]["Authorization"] == "Bearer test_secret_123"
    assert kwargs["params"]["tenantId"] == "tenant-real-1"
    assert kwargs["params"]["agentId"] == "agent-real-2"
    assert kwargs["params"]["agentVersionId"] == "version-real-3"


@pytest.mark.asyncio
async def test_fetch_agent_runtime_missing_secret():
    """Verify RuntimeClientAuthError when secret is missing."""
    client = RuntimeConfigClient(base_url="http://localhost:3000", worker_api_secret=None)
    with pytest.raises(RuntimeClientAuthError):
        await client.fetch_agent_runtime(tenant_id="t1", agent_id="a1")


@pytest.mark.asyncio
async def test_fetch_agent_runtime_401_unauthorized():
    """Verify RuntimeClientAuthError on 401 response."""
    client = RuntimeConfigClient(base_url="http://localhost:3000", worker_api_secret="invalid-secret")

    mock_resp = AsyncMock()
    mock_resp.status = 401
    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_get.__aexit__ = AsyncMock(return_value=None)
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_get)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(RuntimeClientAuthError, match="401 Unauthorized"):
            await client.fetch_agent_runtime(tenant_id="t1", agent_id="a1")


@pytest.mark.asyncio
async def test_fetch_agent_runtime_404_not_found():
    """Verify RuntimeClientNotFoundError on 404 response (cross-tenant or non-existent)."""
    client = RuntimeConfigClient(base_url="http://localhost:3000", worker_api_secret="secret")

    mock_resp = AsyncMock()
    mock_resp.status = 404
    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_get.__aexit__ = AsyncMock(return_value=None)
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_get)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(RuntimeClientNotFoundError, match="404 Not Found"):
            await client.fetch_agent_runtime(tenant_id="t1", agent_id="a1")


@pytest.mark.asyncio
async def test_fetch_agent_runtime_400_unpublished():
    """Verify RuntimeClientValidationError on 400 response (e.g. unpublished version)."""
    client = RuntimeConfigClient(base_url="http://localhost:3000", worker_api_secret="secret")

    mock_resp = AsyncMock()
    mock_resp.status = 400
    mock_resp.json = AsyncMock(return_value={"success": False, "error": {"message": "Version v2 is not published."}})
    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_get.__aexit__ = AsyncMock(return_value=None)
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_get)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(RuntimeClientValidationError, match="not published"):
            await client.fetch_agent_runtime(tenant_id="t1", agent_id="a1")


@pytest.mark.asyncio
async def test_dynamic_prompt_and_greeting_injection(monkeypatch):
    """Verify that participant metadata triggers dynamic prompt & greeting injection in agent entrypoint."""
    from zcloud_ai_telecaller import agent

    custom_prompt = "You are an elite customer support specialist for Acme Corp."
    custom_greeting = "Hello from Acme Corp! How can I make your day great?"

    mock_runtime_config = AgentRuntimeConfig(
        tenant_id="tenant-acme",
        agent_id="agent-support",
        agent_version_id="ver-1",
        name="Acme Agent",
        prompt=custom_prompt,
        greeting=custom_greeting,
        voice_id="en_US-lessac-medium",
    )

    # Mock RuntimeConfigClient
    mock_client = AsyncMock()
    mock_client.fetch_agent_runtime = AsyncMock(return_value=mock_runtime_config)
    monkeypatch.setattr(agent, "RuntimeConfigClient", lambda **kwargs: mock_client)

    # Mock LiveKit context and participant with valid metadata
    mock_participant = MagicMock()
    mock_participant.identity = "test-caller"
    mock_participant.sid = "PA_999"
    mock_participant.metadata = json.dumps({
        "tenantId": "tenant-acme",
        "agentId": "agent-support",
        "agentVersionId": "ver-1",
    })

    mock_ctx = MagicMock()
    mock_ctx.room.name = "acme-test-room"
    mock_ctx.job.id = "job-acme-1"
    mock_ctx.proc.userdata = {}
    mock_ctx.connect = AsyncMock()
    mock_ctx.wait_for_participant = AsyncMock(return_value=mock_participant)
    mock_ctx.add_shutdown_callback = MagicMock()

    mock_session = MagicMock()
    mock_session.on = MagicMock(return_value=lambda fn: fn)
    mock_session.start = AsyncMock()
    mock_session.say = AsyncMock()

    created_agents = []

    def mock_create_agent(instructions=None):
        created_agents.append(instructions)
        return MagicMock(instructions=instructions)

    monkeypatch.setattr(agent, "create_agent", mock_create_agent)
    monkeypatch.setattr(agent, "create_agent_session", lambda **kwargs: mock_session)

    await agent.entrypoint(mock_ctx)

    # 1. Verify dynamic prompt was passed to create_agent
    assert len(created_agents) == 1
    assert created_agents[0] == custom_prompt

    # 2. Verify dynamic greeting was passed to session.say
    mock_session.say.assert_awaited_once_with(custom_greeting)


@pytest.mark.asyncio
async def test_session_aborted_on_control_plane_failure(monkeypatch):
    """Verify that if control plane rejects the request, session.start is NOT called (secure abort)."""
    from zcloud_ai_telecaller import agent

    # Mock RuntimeConfigClient to raise 404 (e.g. cross-tenant attempt)
    mock_client = AsyncMock()
    mock_client.fetch_agent_runtime = AsyncMock(
        side_effect=RuntimeClientNotFoundError("Agent not found for tenant")
    )
    monkeypatch.setattr(agent, "RuntimeConfigClient", lambda **kwargs: mock_client)

    mock_participant = MagicMock()
    mock_participant.identity = "attacker-caller"
    mock_participant.metadata = json.dumps({
        "tenantId": "tenant-attacker",
        "agentId": "agent-victim",
    })

    mock_ctx = MagicMock()
    mock_ctx.room.name = "attack-room"
    mock_ctx.job.id = "job-attack"
    mock_ctx.proc.userdata = {}
    mock_ctx.connect = AsyncMock()
    mock_ctx.wait_for_participant = AsyncMock(return_value=mock_participant)
    mock_ctx.add_shutdown_callback = MagicMock()

    mock_session = MagicMock()
    mock_session.start = AsyncMock()
    mock_session.say = AsyncMock()
    monkeypatch.setattr(agent, "create_agent_session", lambda **kwargs: mock_session)

    await agent.entrypoint(mock_ctx)

    # Verify session was cleanly aborted without starting or speaking
    mock_session.start.assert_not_awaited()
    mock_session.say.assert_not_awaited()

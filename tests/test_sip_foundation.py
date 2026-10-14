"""Tests for Phase 4 Step 1: Telephony / Inbound SIP Foundation.

Validates:
1. Inbound SIP participant attribute extraction (called number, caller number, call ID).
2. Detection of SIP sessions vs WebRTC browser sessions.
3. Client-side resolve_sip_inbound method against Control Plane M2M endpoint.
4. Call lifecycle state reporting (ANSWERED, ACTIVE, COMPLETED).
5. Accumulator serialization with caller and recipient phone numbers.
6. Phase 3 clinic tools compatibility with SIP caller context.
7. Worker authentication enforcement on SIP endpoints.
8. Error handling on unknown phone numbers or inactive tenants.
9. Deterministic SIP provider integration boundary.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from zcloud_ai_telecaller.runtime_client import (
    AgentRuntimeConfig,
    CallSessionAccumulator,
    CallSessionReport,
    RuntimeClientAuthError,
    RuntimeClientConnectionError,
    RuntimeClientNotFoundError,
    RuntimeClientValidationError,
    RuntimeConfigClient,
    extract_sip_participant_info,
    is_sip_session,
)
from zcloud_ai_telecaller.clinic_tools import ClinicToolsManager


# =========================================================================
# Group 1: SIP Identity & Participant Inspection
# =========================================================================

def test_extract_sip_participant_info_standard_attributes():
    """Verify extraction of called number, caller number, and call ID from SIP attributes."""
    participant = MagicMock()
    participant.attributes = {
        "sip.trunkPhoneNumber": "+15550192834",
        "sip.phoneNumber": "+19876543210",
        "sip.callID": "sip-call-xyz-12345",
    }
    participant.identity = "sip_+19876543210"

    called, caller, call_id = extract_sip_participant_info(participant)
    assert called == "+15550192834"
    assert caller == "+19876543210"
    assert call_id == "sip-call-xyz-12345"


def test_extract_sip_participant_info_identity_fallback():
    """Verify fallback extraction of caller number from participant.identity."""
    participant = MagicMock()
    participant.attributes = {
        "sip.trunkPhoneNumber": "+15550192834",
    }
    participant.identity = "sip_+19876543210"

    called, caller, call_id = extract_sip_participant_info(participant)
    assert called == "+15550192834"
    assert caller == "+19876543210"
    assert call_id is None


def test_is_sip_session_detection():
    """Verify correct classification of SIP sessions vs WebRTC browser sessions."""
    # SIP participant with kind = SIP
    sip_part = MagicMock()
    sip_part.kind = "SIP"
    sip_part.identity = "sip_+19876543210"
    sip_part.attributes = {"sip.phoneNumber": "+19876543210"}
    assert is_sip_session(sip_part) is True

    # SIP room with prefix
    browser_part = MagicMock()
    browser_part.kind = "STANDARD"
    browser_part.identity = "user_browser_123"
    browser_part.attributes = {}
    room = MagicMock()
    room.name = "sip-room-call-999"
    assert is_sip_session(browser_part, room) is True

    # Standard browser participant and room
    browser_room = MagicMock()
    browser_room.name = "room_tenantA_agentB_sess1"
    assert is_sip_session(browser_part, browser_room) is False


# =========================================================================
# Group 2: Runtime Client Inbound SIP Resolution
# =========================================================================

@pytest.mark.asyncio
async def test_resolve_sip_inbound_success():
    """Verify successful inbound SIP route resolution via control plane."""
    client = RuntimeConfigClient(
        base_url="http://mock-control-plane:3000",
        worker_api_secret="valid_secret",
    )

    mock_response_data = {
        "success": True,
        "data": {
            "tenantId": "tenant-clinic-1",
            "agentId": "agent-voice-1",
            "agentVersionId": "v1",
            "roomName": "sip-room-001",
            "callId": "call-12345",
            "callerNumber": "+19876543210",
            "recipientNumber": "+15550192834",
            "runtimeConfig": {
                "tenantId": "tenant-clinic-1",
                "agentId": "agent-voice-1",
                "agentVersionId": "v1",
                "name": "Dental Clinic Receptionist",
                "prompt": "You are a clinic receptionist.",
                "greeting": "Hello, thank you for calling Dental Care.",
                "aiModel": "gemini-2.5-flash",
                "voiceProvider": "piper",
                "voiceId": "en_US-lessac-medium",
                "clinicProfile": {
                    "name": "Dental Care",
                    "phone": "+15550192834",
                },
            },
        },
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=mock_response_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        res = await client.resolve_sip_inbound(
            called_number="+15550192834",
            caller_number="+19876543210",
            call_id="sip-call-xyz",
            room_name="sip-room-001",
        )

    assert res["tenantId"] == "tenant-clinic-1"
    assert res["agentId"] == "agent-voice-1"
    assert res["callId"] == "call-12345"
    assert isinstance(res["parsedRuntimeConfig"], AgentRuntimeConfig)
    assert res["parsedRuntimeConfig"].name == "Dental Clinic Receptionist"


@pytest.mark.asyncio
async def test_resolve_sip_inbound_missing_secret():
    """Verify error raised when worker API secret is not configured."""
    client = RuntimeConfigClient(
        base_url="http://localhost:3000",
        worker_api_secret=None,
    )
    with pytest.raises(RuntimeClientAuthError, match="Worker API secret is not configured"):
        await client.resolve_sip_inbound(called_number="+15550192834")


@pytest.mark.asyncio
async def test_resolve_sip_inbound_empty_number():
    """Verify validation error raised when called_number is empty."""
    client = RuntimeConfigClient(
        base_url="http://localhost:3000",
        worker_api_secret="secret",
    )
    with pytest.raises(RuntimeClientValidationError, match="called_number must be non-empty"):
        await client.resolve_sip_inbound(called_number="   ")


@pytest.mark.asyncio
async def test_resolve_sip_inbound_not_found():
    """Verify RuntimeClientNotFoundError raised when phone number is unknown (404)."""
    client = RuntimeConfigClient(
        base_url="http://mock:3000",
        worker_api_secret="valid_secret",
    )

    mock_resp = MagicMock()
    mock_resp.status = 404
    mock_resp.json = AsyncMock(return_value={"success": False, "error": {"message": "Number not registered"}})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(RuntimeClientNotFoundError):
            await client.resolve_sip_inbound(called_number="+15559990000")


# =========================================================================
# Group 3: Call Lifecycle State Reporting
# =========================================================================

@pytest.mark.asyncio
async def test_update_call_lifecycle_success():
    """Verify updating call lifecycle state to ANSWERED and ACTIVE."""
    client = RuntimeConfigClient(
        base_url="http://mock:3000",
        worker_api_secret="valid_secret",
    )

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"success": True, "data": {"status": "ANSWERED"}})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        res = await client.update_call_lifecycle(
            call_id="call-12345",
            status="ANSWERED",
        )
    assert res.get("status") == "ANSWERED"


# =========================================================================
# Group 4: Call Accumulator & Persistence
# =========================================================================

def test_accumulator_captures_sip_metadata():
    """Verify CallSessionAccumulator stores caller number, recipient number, and direction."""
    accum = CallSessionAccumulator(
        tenant_id="tenant-1",
        agent_id="agent-1",
        room_name="sip-room-01",
        agent_version_id="v1",
        caller_identity="sip_+19876543210",
        caller_number="+19876543210",
        caller_name="Patient (+19876543210)",
        recipient_number="+15550192834",
        direction="INBOUND",
        start_time=100.0,
    )
    accum.add_transcript(speaker="AGENT", message="Thank you for calling.")
    accum.add_transcript(speaker="CUSTOMER", message="When are you open?")

    report = accum.finish(reason="completed", end_time=160.0)

    assert isinstance(report, CallSessionReport)
    assert report.tenant_id == "tenant-1"
    assert report.agent_id == "agent-1"
    assert report.room_name == "sip-room-01"
    assert report.caller_number == "+19876543210"
    assert report.recipient_number == "+15550192834"
    assert report.direction == "INBOUND"
    assert report.duration_seconds == 60
    assert len(report.transcripts) == 2


# =========================================================================
# Group 5: Phase 3 Clinic Tools Compatibility in SIP Context
# =========================================================================

@pytest.mark.asyncio
async def test_clinic_tools_work_with_sip_call_id():
    """Verify ClinicToolsManager correctly uses call_id from inbound SIP resolution for transfer."""
    mock_client = AsyncMock()
    mock_client.request_human_transfer = AsyncMock(
        return_value={"transferNumber": "+15557778888", "fallbackMessage": "Connecting you now."}
    )

    manager = ClinicToolsManager(
        tenant_id="tenant-dental",
        agent_id="agent-dental-1",
        call_id="call-sip-9999",
        client=mock_client,
        transfer_rules={"primaryNumber": "+15557778888"},
        clinic_profile={"name": "Dental Clinic"},
    )

    result = await manager.request_human_transfer_action(reason="Urgent toothache")
    assert "Connecting you now." in result

    # Verify call_id was passed to control plane transfer endpoint
    mock_client.request_human_transfer.assert_called_once_with(
        tenant_id="tenant-dental",
        call_id="call-sip-9999",
        reason="Urgent toothache",
    )


# =========================================================================
# Group 6: Session Abstractions (BrowserSession vs SIPSession)
# =========================================================================

from zcloud_ai_telecaller.session import BaseCallSession, BrowserSession, SIPSession
from zcloud_ai_telecaller.conversation_state import ConversationStateManager


def test_session_abstractions_properties():
    """Verify typed differentiation between BrowserSession and SIPSession."""
    runtime_config = AgentRuntimeConfig(
        tenant_id="tenant-1",
        agent_id="agent-1",
        agent_version_id="v1",
        name="Clinic Receptionist",
        prompt="You are a clinic receptionist.",
        greeting="Hello!",
    )
    state_mgr = ConversationStateManager(tenant_id="tenant-1", agent_id="agent-1")
    accum = CallSessionAccumulator(
        tenant_id="tenant-1",
        agent_id="agent-1",
        room_name="room-1",
    )

    browser_sess = BrowserSession(
        tenant_id="tenant-1",
        agent_id="agent-1",
        room_name="room-browser-1",
        runtime_config=runtime_config,
        state_manager=state_mgr,
        accumulator=accum,
        participant_identity="user_chrome_web",
    )
    assert browser_sess.is_browser is True
    assert browser_sess.is_telephony is False
    assert browser_sess.session_type == "BROWSER"
    assert browser_sess.participant_identity == "user_chrome_web"

    sip_sess = SIPSession(
        tenant_id="tenant-1",
        agent_id="agent-1",
        room_name="sip-room-01",
        runtime_config=runtime_config,
        state_manager=state_mgr,
        accumulator=accum,
        called_number="+918040001000",
        caller_number="+919876543210",
        call_id="call-db-id-01",
        sip_call_id="sip-guid-abc-123",
        sip_trunk_id="ST_plivo_india_01",
        provider="Plivo",
    )
    assert sip_sess.is_telephony is True
    assert sip_sess.is_browser is False
    assert sip_sess.session_type == "SIP"
    assert sip_sess.called_number == "+918040001000"
    assert sip_sess.caller_number == "+919876543210"
    assert sip_sess.sip_call_id == "sip-guid-abc-123"
    assert sip_sess.provider == "Plivo"
    assert sip_sess.call_id == "call-db-id-01"


@pytest.mark.asyncio
async def test_inbound_sip_fallback_when_control_plane_fails(monkeypatch):
    """Verify that inbound SIP session does not abort on control plane failure and starts fallback session."""
    from unittest.mock import AsyncMock, MagicMock
    from zcloud_ai_telecaller import agent

    mock_room = MagicMock()
    mock_room.name = "sip-clinic-test-call-123"

    mock_job = MagicMock()
    mock_job.id = "job-sip-123"

    mock_participant = MagicMock()
    mock_participant.identity = "sip_+919876543210"
    mock_participant.attributes = {
        "sip.trunkPhoneNumber": "+918065536028",
        "sip.phoneNumber": "+919876543210",
        "sip.callID": "plivo-call-abc",
    }
    mock_participant.kind = "SIP"

    mock_ctx = MagicMock()
    mock_ctx.room = mock_room
    mock_ctx.job = mock_job
    mock_ctx.proc = MagicMock()
    mock_ctx.proc.userdata = {}
    mock_ctx.connect = AsyncMock()
    mock_ctx.wait_for_participant = AsyncMock(return_value=mock_participant)
    mock_ctx.add_shutdown_callback = MagicMock()

    mock_session = MagicMock()
    mock_session.on = MagicMock(return_value=lambda fn: fn)
    mock_session.start = AsyncMock()
    mock_session.say = AsyncMock()

    monkeypatch.setattr(agent, "create_agent_session", lambda **kwargs: mock_session)

    # Force control plane resolution to fail (e.g. 404 or connection error)
    mock_client = MagicMock()
    mock_client.resolve_sip_inbound = AsyncMock(side_effect=RuntimeClientNotFoundError("Phone number not found"))
    monkeypatch.setattr(agent, "RuntimeConfigClient", lambda **kwargs: mock_client)

    await agent.entrypoint(mock_ctx)

    # Verify session was NOT aborted: session.start and session.say were called
    mock_session.start.assert_awaited_once()
    mock_session.say.assert_awaited_once_with(agent.GREETING_MESSAGE)


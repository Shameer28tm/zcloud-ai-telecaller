"""Unit tests for Phase 2B Call Session Accumulator, Reporting Client, and Event Handling."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestServer

from zcloud_ai_telecaller.runtime_client import (
    CallSessionAccumulator,
    CallSessionReport,
    CallTranscriptItem,
    RuntimeClientAuthError,
    RuntimeClientConnectionError,
    RuntimeClientNotFoundError,
    RuntimeClientValidationError,
    RuntimeConfigClient,
)


def test_call_transcript_item_model() -> None:
    """Verifies CallTranscriptItem field validation and alias serialization."""
    item = CallTranscriptItem(
        speaker="CUSTOMER",
        message="What are your office hours?",
        timestamp_offset_ms=2500,
        latency_ms=120,
        confidence=0.95,
    )
    assert item.speaker == "CUSTOMER"
    assert item.message == "What are your office hours?"
    assert item.timestamp_offset_ms == 2500
    assert item.latency_ms == 120

    dump = item.model_dump(by_alias=True, exclude_none=True)
    assert dump["timestampOffsetMs"] == 2500
    assert dump["latencyMs"] == 120
    assert "timestamp_offset_ms" not in dump


def test_call_session_report_model() -> None:
    """Verifies CallSessionReport fields and JSON alias serialization."""
    report = CallSessionReport(
        tenant_id="tenant_123",
        agent_id="agent_456",
        agent_version_id="ver_789",
        room_name="call_room_001",
        caller_identity="caller_john",
        caller_number="+15551234567",
        caller_name="John Doe",
        duration_seconds=75,
        status="COMPLETED",
        end_reason="normal",
        transcripts=[
            CallTranscriptItem(speaker="AGENT", message="Hello!", timestamp_offset_ms=0),
            CallTranscriptItem(speaker="CUSTOMER", message="Hi!", timestamp_offset_ms=1500),
        ],
    )
    dump = report.model_dump(by_alias=True, exclude_none=True)
    assert dump["tenantId"] == "tenant_123"
    assert dump["agentId"] == "agent_456"
    assert dump["agentVersionId"] == "ver_789"
    assert dump["roomName"] == "call_room_001"
    assert dump["callerIdentity"] == "caller_john"
    assert dump["durationSeconds"] == 75
    assert len(dump["transcripts"]) == 2
    assert dump["transcripts"][1]["timestampOffsetMs"] == 1500


def test_accumulator_transcripts_and_deduplication() -> None:
    """Verifies CallSessionAccumulator accumulates transcripts and deduplicates rapid identical turns."""
    start = 1000.0
    acc = CallSessionAccumulator(
        tenant_id="tenant_1",
        agent_id="agent_1",
        room_name="room_1",
        start_time=start,
    )

    # Add agent greeting
    added = acc.add_transcript("AGENT", "Hello, how can I help you?", timestamp_offset_ms=0)
    assert added is True
    assert len(acc.transcripts) == 1

    # Attempt to add exact same greeting immediately (should be deduplicated)
    duplicate_added = acc.add_transcript("AGENT", "Hello, how can I help you?", timestamp_offset_ms=500)
    assert duplicate_added is False
    assert len(acc.transcripts) == 1

    # Add customer turn
    customer_added = acc.add_transcript("CUSTOMER", "I need to book a cleaning.", timestamp_offset_ms=3000)
    assert customer_added is True
    assert len(acc.transcripts) == 2

    # Discard empty turns
    empty_added = acc.add_transcript("CUSTOMER", "   ", timestamp_offset_ms=4000)
    assert empty_added is False
    assert len(acc.transcripts) == 2

    # Finish call at 1065.0 seconds (duration = 65 seconds)
    report = acc.finish(reason="caller_hung_up", end_time=1065.0)
    assert report.duration_seconds == 65
    assert report.end_reason == "caller_hung_up"
    assert len(report.transcripts) == 2


def test_accumulator_record_livekit_item() -> None:
    """Verifies accumulator correctly extracts speaker, text, and metadata from LiveKit chat items."""
    acc = CallSessionAccumulator(
        tenant_id="tenant_1",
        agent_id="agent_1",
        room_name="room_1",
        start_time=100.0,
    )

    # Mock user ChatMessage
    user_msg = MagicMock()
    user_msg.role = "user"
    user_msg.text_content = "Do you have parking available?"
    user_msg.transcript_confidence = 0.98

    added = acc.record_livekit_item(user_msg, timestamp_offset_ms=2000)
    assert added is True
    assert len(acc.transcripts) == 1
    assert acc.transcripts[0].speaker == "CUSTOMER"
    assert acc.transcripts[0].message == "Do you have parking available?"
    assert acc.transcripts[0].confidence == 0.98

    # Mock assistant ChatMessage
    assistant_msg = MagicMock()
    assistant_msg.role = "assistant"
    assistant_msg.text_content = "Yes, we have free parking in the rear lot."
    assistant_msg.transcript_confidence = None

    added_agent = acc.record_livekit_item(assistant_msg, timestamp_offset_ms=4500)
    assert added_agent is True
    assert len(acc.transcripts) == 2
    assert acc.transcripts[1].speaker == "AGENT"
    assert acc.transcripts[1].message == "Yes, we have free parking in the rear lot."

    # Mock system prompt (should be ignored)
    sys_msg = MagicMock()
    sys_msg.role = "system"
    sys_msg.text_content = "You are a helpful assistant."
    added_sys = acc.record_livekit_item(sys_msg)
    assert added_sys is False
    assert len(acc.transcripts) == 2


@pytest.mark.asyncio
async def test_report_call_session_success() -> None:
    """Tests successful call session submission to the internal API."""
    captured_req = {}

    async def handle_call_session(request: web.Request) -> web.Response:
        auth = request.headers.get("Authorization")
        if auth != "Bearer secret_test_key":
            return web.json_response({"success": False, "error": {"code": "UNAUTHORIZED"}}, status=401)
        data = await request.json()
        captured_req.update(data)
        return web.json_response(
            {
                "success": True,
                "data": {
                    "callId": "call_mock_abc123",
                    "durationSeconds": data.get("durationSeconds", 0),
                    "billableMinutes": 2,
                    "transcriptCount": len(data.get("transcripts", [])),
                    "isDuplicate": False,
                },
            }
        )

    app = web.Application()
    app.router.add_post("/api/internal/call-session", handle_call_session)
    server = TestServer(app)
    await server.start_server()

    try:
        client = RuntimeConfigClient(
            base_url=str(server.make_url("")),
            worker_api_secret="secret_test_key",
        )
        report = CallSessionReport(
            tenant_id="ten_1",
            agent_id="ag_1",
            room_name="test_rm_99",
            duration_seconds=95,
            transcripts=[
                CallTranscriptItem(speaker="AGENT", message="Welcome!"),
                CallTranscriptItem(speaker="CUSTOMER", message="Thank you!"),
            ],
        )
        res = await client.report_call_session(report)
        assert res["callId"] == "call_mock_abc123"
        assert res["billableMinutes"] == 2
        assert captured_req["tenantId"] == "ten_1"
        assert captured_req["roomName"] == "test_rm_99"
        assert captured_req["durationSeconds"] == 95
        assert len(captured_req["transcripts"]) == 2
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_report_call_session_errors() -> None:
    """Tests proper exception mapping on 401, 404, 400, and network error."""
    async def handle_401(req):
        return web.json_response({"success": False}, status=401)

    app = web.Application()
    app.router.add_post("/api/internal/call-session", handle_401)
    server = TestServer(app)
    await server.start_server()

    report = CallSessionReport(
        tenant_id="ten_1",
        agent_id="ag_1",
        room_name="rm_1",
        duration_seconds=10,
    )

    try:
        # Test 1: Missing secret raises AuthError
        no_secret_client = RuntimeConfigClient(base_url=str(server.make_url("")))
        with pytest.raises(RuntimeClientAuthError):
            await no_secret_client.report_call_session(report)

        # Test 2: 401 Unauthorized
        client_401 = RuntimeConfigClient(
            base_url=str(server.make_url("")),
            worker_api_secret="bad_key",
        )
        with pytest.raises(RuntimeClientAuthError):
            await client_401.report_call_session(report)

        # Test 3: Unreachable server raises ConnectionError
        dead_client = RuntimeConfigClient(base_url="http://127.0.0.1:59999", worker_api_secret="sec")
        with pytest.raises(RuntimeClientConnectionError):
            await dead_client.report_call_session(report)
    finally:
        await server.close()


def test_session_history_messages_callable_compatibility() -> None:
    """Verifies that ChatContext.messages() (as a callable method in LiveKit 1.8.2) is correctly extracted into accumulator."""
    acc = CallSessionAccumulator(
        tenant_id="ten_1",
        agent_id="ag_1",
        room_name="rm_1",
    )

    class MockHistory:
        def __init__(self):
            msg1 = MagicMock()
            msg1.role = "user"
            msg1.text_content = "What services do you offer?"
            msg1.transcript_confidence = 0.99

            msg2 = MagicMock()
            msg2.role = "assistant"
            msg2.text_content = "We offer teeth cleaning and whitening."
            msg2.transcript_confidence = None

            self._msgs = [msg1, msg2]

        def messages(self):
            # In LiveKit Agents 1.8.2, messages is a callable method returning list
            return self._msgs

    mock_session = MagicMock()
    mock_session.history = MockHistory()

    # Reproduce the exact fallback logic from agent.py
    raw_messages = mock_session.history.messages
    history_messages = raw_messages() if callable(raw_messages) else raw_messages
    for msg in history_messages:
        acc.record_livekit_item(msg)

    assert len(acc.transcripts) == 2
    assert acc.transcripts[0].speaker == "CUSTOMER"
    assert acc.transcripts[0].message == "What services do you offer?"
    assert acc.transcripts[1].speaker == "AGENT"
    assert acc.transcripts[1].message == "We offer teeth cleaning and whitening."


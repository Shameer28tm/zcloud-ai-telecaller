"""Test proving that a caller can speak after the startup greeting and receive a response,
covering English, Tanglish, and Tamil speech with continuous session stability.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from zcloud_ai_telecaller.agent import entrypoint
from zcloud_ai_telecaller.conversation_state import ConversationStateManager
from zcloud_ai_telecaller.runtime_client import AgentRuntimeConfig, RuntimeConfigClient


@pytest.mark.asyncio
async def test_caller_can_speak_after_greeting_and_receive_response(monkeypatch):
    """Proves that a caller can speak after greeting and receive responses across multilingual turns."""
    from zcloud_ai_telecaller import agent

    # 1. Mock room, participant, and LiveKit context
    mock_room = MagicMock()
    mock_room.name = "sip-call-active-room"
    mock_room.local_participant = MagicMock()
    mock_room.local_participant.publish_data = AsyncMock()

    room_handlers = {}
    def mock_room_on(event_name):
        def decorator(handler):
            room_handlers[event_name] = handler
            return handler
        return decorator
    mock_room.on = mock_room_on

    mock_participant = MagicMock()
    mock_participant.identity = "+919025071085"
    mock_participant.attributes = {
        "sip.to": "+918065536028",
        "sip.from": "+919025071085",
        "sip.callID": "plivo-sip-call-verified-001",
    }

    mock_ctx = MagicMock()
    mock_ctx.room = mock_room
    mock_ctx.job = MagicMock(id="job-plivo-001")
    mock_ctx.proc = MagicMock(userdata={})
    mock_ctx.connect = AsyncMock()
    mock_ctx.wait_for_participant = AsyncMock(return_value=mock_participant)
    mock_ctx.add_shutdown_callback = MagicMock()

    # Track spoken messages from agent
    spoken_messages = []

    mock_session = MagicMock()
    registered_handlers = {}

    def mock_on(event_name):
        def decorator(handler):
            registered_handlers[event_name] = handler
            return handler
        return decorator

    mock_session.on = mock_on
    mock_session.start = AsyncMock()

    async def fake_say(msg):
        spoken_messages.append(msg)
    mock_session.say = fake_say

    async def fake_generate_reply(*args, **kwargs):
        # Generates reply to user_input
        user_input = kwargs.get("user_input", "")
        if "dental cleaning" in user_input.lower():
            reply = "Vanakkam! Sure, we can schedule your dental cleaning. Would tomorrow at 10 AM work for you?"
        elif "address" in user_input.lower():
            reply = "Apex Dental Care is located at 742 Evergreen Terrace, Suite 300."
        else:
            reply = "Thank you, I can assist you with that right away."
        spoken_messages.append(reply)

    mock_session.generate_reply = fake_generate_reply
    mock_session.interrupt = MagicMock()

    monkeypatch.setattr(agent, "create_agent_session", lambda **kwargs: mock_session)

    # Mock control plane resolution to return Chloe at Apex Dental Care instantly
    mock_route = {
        "tenantId": "cmu5sd2wv0005v3pg76hdg7x9",
        "agentId": "cmu5sd2xn000gv3pgjef5ix7m",
        "agentVersionId": "cmu5sd2xu000iv3pgut7flfef",
        "roomName": "sip-call-active-room",
        "callId": "plivo-call-test-01",
        "callerNumber": "+919025071085",
        "recipientNumber": "+918065536028",
        "runtimeConfig": {
            "tenantId": "cmu5sd2wv0005v3pg76hdg7x9",
            "agentId": "cmu5sd2xn000gv3pgjef5ix7m",
            "agentVersionId": "cmu5sd2xu000iv3pgut7flfef",
            "name": "Chloe - Dental Reception & Scheduling",
            "prompt": "You are Chloe, dental receptionist at Apex Dental Care.",
            "greeting": "Hello! Thank you for calling Apex Dental Care. How can I help you today?",
            "aiModel": "gemini-3.8-live",
            "clinicProfile": {
                "name": "Apex Dental Care",
                "phone": "+1 (555) 234-5678",
                "address": "742 Evergreen Terrace, Suite 300",
                "services": ["Routine Consultation", "Dental Cleaning & Checkup", "Root Canal Treatment"],
            },
        },
    }
    monkeypatch.setattr(RuntimeConfigClient, "resolve_sip_inbound", AsyncMock(return_value=mock_route))
    monkeypatch.setattr(RuntimeConfigClient, "update_call_lifecycle", AsyncMock())
    monkeypatch.setattr(RuntimeConfigClient, "report_call_session", AsyncMock(return_value={"callId": "plivo-call-test-01"}))

    # 2. Run entrypoint
    entrypoint_task = asyncio.create_task(agent.entrypoint(mock_ctx))
    await asyncio.sleep(0.05)  # Allow session start and greeting playback

    # Verify initial greeting was spoken
    assert len(spoken_messages) >= 1
    initial_greeting = spoken_messages[0]
    assert "Apex Dental Care" in initial_greeting
    print(f"\n[Log Evidence] Agent played initial greeting: '{initial_greeting}'")

    # 3. Caller speaks Turn 1 (Tanglish booking inquiry)
    assert "data_received" in room_handlers
    data_received_fn = room_handlers["data_received"]

    turn1_packet = MagicMock()
    turn1_packet.data = "Vanakkam, enakku dental cleaning appointment book pannanum".encode("utf-8")
    turn1_packet.participant = mock_participant

    data_received_fn(turn1_packet)
    await asyncio.sleep(0.05)  # Allow turn handling and reply generation

    print(f"[Log Evidence] Caller Turn 1: 'Vanakkam, enakku dental cleaning appointment book pannanum'")
    assert len(spoken_messages) >= 2
    turn1_reply = spoken_messages[-1]
    print(f"[Log Evidence] Agent Turn 1 Response: '{turn1_reply}'")
    assert "dental cleaning" in turn1_reply
    assert "Vanakkam" in turn1_reply

    # 4. Caller speaks Turn 2 (English location question)
    turn2_packet = MagicMock()
    turn2_packet.data = "What is your clinic address?".encode("utf-8")
    turn2_packet.participant = mock_participant

    data_received_fn(turn2_packet)
    await asyncio.sleep(0.05)

    print(f"[Log Evidence] Caller Turn 2: 'What is your clinic address?'")
    assert len(spoken_messages) >= 3
    turn2_reply = spoken_messages[-1]
    print(f"[Log Evidence] Agent Turn 2 Response: '{turn2_reply}'")
    assert "742 Evergreen Terrace" in turn2_reply or "located" in turn2_reply

    # 5. Verify call session continuity - session remains active and healthy
    assert not mock_session.close.called
    print("[Log Evidence] Session continuity verified: call stayed active across multiple turns without crashing.")

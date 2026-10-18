"""Unit and integration tests for CallSessionState state machine and idempotency guarantees."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from zcloud_ai_telecaller.conversation_state import ConversationStateManager
from zcloud_ai_telecaller.runtime_client import (
    AgentRuntimeConfig,
    CallSessionAccumulator,
    RuntimeConfigClient,
)
from zcloud_ai_telecaller.session import (
    BaseCallSession,
    BrowserSession,
    CallSessionState,
    SIPSession,
)
from zcloud_ai_telecaller.clinic_tools import ClinicToolsManager


class TestCallSessionStateMachine(unittest.TestCase):
    """Verifies state machine transitions, terminal state protection, and history tracking."""

    def setUp(self) -> None:
        self.runtime_config = AgentRuntimeConfig(
            tenant_id="test-tenant",
            agent_id="test-agent",
            agent_version_id="ver-1",
            name="Test Assistant",
            prompt="You are helpful.",
            greeting="Hello!",
        )
        self.state_manager = ConversationStateManager("test-tenant", "test-agent")
        self.accumulator = CallSessionAccumulator(
            tenant_id="test-tenant",
            agent_id="test-agent",
            room_name="test-room",
        )
        self.session = BaseCallSession(
            session_type="SIP",
            tenant_id="test-tenant",
            agent_id="test-agent",
            room_name="test-room",
            runtime_config=self.runtime_config,
            state_manager=self.state_manager,
            accumulator=self.accumulator,
            call_id="call-12345",
        )

    def test_initial_state_is_created(self) -> None:
        self.assertEqual(self.session.state, CallSessionState.CREATED)
        self.assertFalse(self.session.is_active)
        self.assertFalse(self.session.is_terminal)
        self.assertEqual(len(self.session.state_transitions), 1)
        self.assertEqual(self.session.state_transitions[0]["toState"], "CREATED")

    def test_valid_lifecycle_transitions(self) -> None:
        # CREATED -> CONNECTING
        ok = self.session.transition_to(CallSessionState.CONNECTING, "sip_invite_received")
        self.assertTrue(ok)
        self.assertEqual(self.session.state, CallSessionState.CONNECTING)
        self.assertFalse(self.session.is_active)
        self.assertFalse(self.session.is_terminal)

        # CONNECTING -> ACTIVE
        ok = self.session.transition_to(CallSessionState.ACTIVE, "caller_answered")
        self.assertTrue(ok)
        self.assertEqual(self.session.state, CallSessionState.ACTIVE)
        self.assertTrue(self.session.is_active)
        self.assertFalse(self.session.is_terminal)

        # ACTIVE -> COMPLETED
        ok = self.session.transition_to(CallSessionState.COMPLETED, "normal_hangup")
        self.assertTrue(ok)
        self.assertEqual(self.session.state, CallSessionState.COMPLETED)
        self.assertFalse(self.session.is_active)
        self.assertTrue(self.session.is_terminal)

    def test_idempotent_same_state_transition(self) -> None:
        self.session.transition_to(CallSessionState.CONNECTING)
        initial_history_len = len(self.session.state_transitions)

        # Transitioning to CONNECTING again should be a no-op
        ok = self.session.transition_to(CallSessionState.CONNECTING)
        self.assertTrue(ok)
        self.assertEqual(self.session.state, CallSessionState.CONNECTING)
        self.assertEqual(len(self.session.state_transitions), initial_history_len)

    def test_terminal_protection_from_completed(self) -> None:
        self.session.transition_to(CallSessionState.ACTIVE)
        self.session.transition_to(CallSessionState.COMPLETED, "call_ended")
        self.assertTrue(self.session.is_terminal)

        # Attempting to transition from COMPLETED back to ACTIVE must be rejected
        ok = self.session.transition_to(CallSessionState.ACTIVE, "rogue_reconnect")
        self.assertFalse(ok)
        self.assertEqual(self.session.state, CallSessionState.COMPLETED)

    def test_terminal_protection_from_failed(self) -> None:
        self.session.transition_to(CallSessionState.FAILED, "fatal_audio_error")
        self.assertTrue(self.session.is_terminal)

        ok = self.session.transition_to(CallSessionState.ACTIVE)
        self.assertFalse(ok)
        self.assertEqual(self.session.state, CallSessionState.FAILED)

    def test_terminal_protection_from_cancelled(self) -> None:
        self.session.transition_to(CallSessionState.CANCELLED, "caller_cancelled")
        self.assertTrue(self.session.is_terminal)

        ok = self.session.transition_to(CallSessionState.ACTIVE)
        self.assertFalse(ok)
        self.assertEqual(self.session.state, CallSessionState.CANCELLED)

    def test_string_state_compatibility(self) -> None:
        ok = self.session.transition_to("CONNECTING", "string_transition")
        self.assertTrue(ok)
        self.assertEqual(self.session.state, CallSessionState.CONNECTING)


class TestBookingIdempotencyForwarding(unittest.IsolatedAsyncioTestCase):
    """Verifies that book_appointment constructs and forwards explicit idempotency keys."""

    async def test_runtime_client_payload_includes_idempotency_key(self) -> None:
        client = RuntimeConfigClient(
            base_url="http://localhost:3000",
            worker_api_secret="test_secret_12345678",
        )

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"success": True, "data": {"appointmentId": "apt_999"}})

        mock_post = MagicMock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            res = await client.book_appointment(
                tenant_id="tenant_abc",
                patient_name="John Doe",
                phone="+15551234567",
                date="2026-10-25",
                time="10:30",
                call_id="call_999",
                idempotency_key="call_999_+15551234567_2026-10-25_10:30",
            )

            self.assertEqual(res["appointmentId"], "apt_999")
            call_args = mock_session.post.call_args
            payload = call_args[1]["json"]
            self.assertEqual(payload["idempotencyKey"], "call_999_+15551234567_2026-10-25_10:30")
            self.assertEqual(payload["callId"], "call_999")

    async def test_clinic_tools_passes_deterministic_idempotency_key(self) -> None:
        mock_client = AsyncMock()
        mock_client.book_appointment = AsyncMock(return_value={"appointmentId": "apt_confirmed_1", "isExisting": False})

        manager = ClinicToolsManager(
            tenant_id="tenant_apex",
            agent_id="agent_dr_reed",
            call_id="call_session_777",
            client=mock_client,
            clinic_profile={"name": "Apex Clinic"},
        )

        result = await manager.book_appointment_action(
            patient_name="Jane Smith",
            phone="555-987-6543",
            date="2026-10-28",
            time="14:00",
            service="Dental Cleaning",
        )

        self.assertIn("scheduled", result.lower())
        mock_client.book_appointment.assert_awaited_once()
        kwargs = mock_client.book_appointment.call_args[1]
        self.assertEqual(kwargs["idempotency_key"], "call_session_777_+15559876543_2026-10-28_14:00")


if __name__ == "__main__":
    unittest.main()

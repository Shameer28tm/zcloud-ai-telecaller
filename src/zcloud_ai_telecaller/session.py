"""Authoritative Session Abstractions for Browser WebRTC and Telephony SIP Calls."""

from __future__ import annotations

from enum import Enum
import logging
import time
from typing import Any, Optional

from zcloud_ai_telecaller.runtime_client import (
    AgentRuntimeConfig,
    CallSessionAccumulator,
)
from zcloud_ai_telecaller.conversation_state import ConversationStateManager

logger = logging.getLogger("zcloud_ai_telecaller.session")


class CallSessionState(str, Enum):
    """Authoritative lifecycle states for an AI voice session."""

    CREATED = "CREATED"
    CONNECTING = "CONNECTING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class BaseCallSession:
    """Base call session representing an active interaction between a caller and the AI."""

    def __init__(
        self,
        session_type: str,
        tenant_id: str,
        agent_id: str,
        room_name: str,
        runtime_config: AgentRuntimeConfig,
        state_manager: ConversationStateManager,
        accumulator: CallSessionAccumulator,
        agent_version_id: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> None:
        self.session_type = session_type
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.room_name = room_name
        self.runtime_config = runtime_config
        self.state_manager = state_manager
        self.accumulator = accumulator
        self.agent_version_id = agent_version_id
        self.call_id = call_id

        # Explicit Call Lifecycle State Machine
        self._state: CallSessionState = CallSessionState.CREATED
        self.state_transitions: list[dict[str, Any]] = [
            {
                "fromState": None,
                "toState": CallSessionState.CREATED.value,
                "timestamp": time.time(),
                "reason": "initialization",
            }
        ]

    @property
    def state(self) -> CallSessionState:
        """Returns the current authoritative lifecycle state of this call."""
        return self._state

    @property
    def is_terminal(self) -> bool:
        """Returns True if the call has reached a terminal finalization state."""
        return self._state in (
            CallSessionState.COMPLETED,
            CallSessionState.FAILED,
            CallSessionState.CANCELLED,
        )

    @property
    def is_active(self) -> bool:
        """Returns True if the call is currently in an active conversational turn state."""
        return self._state == CallSessionState.ACTIVE

    def transition_to(
        self,
        new_state: CallSessionState | str,
        reason: Optional[str] = None,
    ) -> bool:
        """Transitions the call to a new lifecycle state with unidirectional validation.

        Terminal states (COMPLETED, FAILED, CANCELLED) cannot transition back to active states.
        Idempotent: Transitioning to the same state is safe and accepted without error.
        """
        target = CallSessionState(new_state) if isinstance(new_state, str) else new_state

        if self._state == target:
            return True

        # Unidirectional terminal guard: cannot transition out of terminal states
        if self.is_terminal:
            logger.warning(
                "Rejected invalid state transition for call %s: cannot transition from terminal '%s' to '%s'",
                self.call_id or self.room_name,
                self._state.value,
                target.value,
            )
            return False

        old_state = self._state
        self._state = target
        self.state_transitions.append({
            "fromState": old_state.value,
            "toState": target.value,
            "timestamp": time.time(),
            "reason": reason or "normal_lifecycle",
        })
        logger.info(
            "CallSession (%s) state transition: %s -> %s (reason: %s)",
            self.call_id or self.room_name,
            old_state.value,
            target.value,
            reason or "normal_lifecycle",
        )
        return True

    @property
    def is_telephony(self) -> bool:
        """Returns True if this session originated from PSTN/SIP telephony."""
        return self.session_type.upper() == "SIP"

    @property
    def is_browser(self) -> bool:
        """Returns True if this session originated from browser WebRTC."""
        return self.session_type.upper() == "BROWSER"


class BrowserSession(BaseCallSession):
    """Browser-based WebRTC voice session."""

    def __init__(
        self,
        tenant_id: str,
        agent_id: str,
        room_name: str,
        runtime_config: AgentRuntimeConfig,
        state_manager: ConversationStateManager,
        accumulator: CallSessionAccumulator,
        participant_identity: str,
        agent_version_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            session_type="BROWSER",
            tenant_id=tenant_id,
            agent_id=agent_id,
            room_name=room_name,
            runtime_config=runtime_config,
            state_manager=state_manager,
            accumulator=accumulator,
            agent_version_id=agent_version_id,
        )
        self.participant_identity = participant_identity
        self.user_id = user_id


class SIPSession(BaseCallSession):
    """PSTN / Carrier SIP Inbound Telephony Session."""

    def __init__(
        self,
        tenant_id: str,
        agent_id: str,
        room_name: str,
        runtime_config: AgentRuntimeConfig,
        state_manager: ConversationStateManager,
        accumulator: CallSessionAccumulator,
        called_number: str,
        call_id: str,
        caller_number: Optional[str] = None,
        sip_call_id: Optional[str] = None,
        sip_trunk_id: Optional[str] = None,
        provider: str = "Plivo",
        agent_version_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            session_type="SIP",
            tenant_id=tenant_id,
            agent_id=agent_id,
            room_name=room_name,
            runtime_config=runtime_config,
            state_manager=state_manager,
            accumulator=accumulator,
            agent_version_id=agent_version_id,
            call_id=call_id,
        )
        self.called_number = called_number
        self.caller_number = caller_number
        self.sip_call_id = sip_call_id
        self.sip_trunk_id = sip_trunk_id
        self.provider = provider

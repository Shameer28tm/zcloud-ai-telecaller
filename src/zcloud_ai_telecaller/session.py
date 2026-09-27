"""Authoritative Session Abstractions for Browser WebRTC and Telephony SIP Calls."""

from __future__ import annotations

import logging
from typing import Any, Optional

from zcloud_ai_telecaller.runtime_client import (
    AgentRuntimeConfig,
    CallSessionAccumulator,
)
from zcloud_ai_telecaller.conversation_state import ConversationStateManager

logger = logging.getLogger("zcloud_ai_telecaller.session")


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

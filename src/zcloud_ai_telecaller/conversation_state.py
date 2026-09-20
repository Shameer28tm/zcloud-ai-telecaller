"""Structured conversation state tracking and token budget manager for Z Cloud AI Telecaller."""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any, Optional

logger = logging.getLogger("zcloud_ai_telecaller.conversation_state")


class ConversationStateManager:
    """Maintains authoritative structured session state, slot memory, and bounded token context."""

    def __init__(
        self,
        tenant_id: str,
        agent_id: str,
        max_context_messages: int = 10,
        max_output_tokens: int = 250,
    ) -> None:
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.max_context_messages = max_context_messages
        self.max_output_tokens = max_output_tokens

        # State fields
        self.current_intent: str = "GENERAL_INQUIRY"
        self.requested_date: Optional[str] = None
        self.requested_time: Optional[str] = None
        self.service_name: str = "General Consultation"
        self.available_slots: list[str] = []
        self.selected_slot: Optional[str] = None
        self.patient_name: Optional[str] = None
        self.patient_phone: Optional[str] = None
        self.is_awaiting_confirmation: bool = False
        self.appointment_id: Optional[str] = None
        self.booking_confirmed: bool = False

        # Metrics and lifecycle
        self.start_timestamp: float = time.time()
        self.turn_count: int = 0
        self.tool_call_count: int = 0
        self.messages: list[dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Appends a conversation message and enforces bounded context window to minimize LLM token usage."""
        self.messages.append({"role": role, "content": content})
        self.turn_count += 1

        # Bounded context window: keep only the most recent N messages
        if len(self.messages) > self.max_context_messages:
            trimmed_count = len(self.messages) - self.max_context_messages
            logger.debug("Context window budget reached. Trimming %d older message(s).", trimmed_count)
            self.messages = self.messages[-self.max_context_messages:]

    def record_availability_result(self, date: str, slots: list[str], service: str = "General Consultation") -> None:
        """Authoritatively stores available slots from backend tool to prevent hallucinations."""
        self.requested_date = date
        self.available_slots = list(slots)
        self.service_name = service
        self.current_intent = "CHECK_AVAILABILITY"

    def is_slot_available(self, time_str: str) -> bool:
        """Verifies whether a requested time exists in backend-verified available slots."""
        clean_time = time_str.strip()
        return clean_time in self.available_slots

    def set_pending_booking(
        self,
        slot: str,
        patient_name: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> None:
        """Sets selected slot and flags that confirmation is required."""
        self.selected_slot = slot
        if patient_name:
            self.patient_name = patient_name.strip()
        if phone:
            self.patient_phone = phone.strip()
        self.is_awaiting_confirmation = True
        self.current_intent = "CONFIRM_BOOKING"

    def mark_booking_confirmed(self, appointment_id: str) -> None:
        """Marks booking as successfully confirmed."""
        self.appointment_id = appointment_id
        self.booking_confirmed = True
        self.is_awaiting_confirmation = False
        self.current_intent = "BOOKING_COMPLETED"

    def get_structured_summary(self) -> dict[str, Any]:
        """Provides a compact summary of active state for prompt context injection."""
        return {
            "intent": self.current_intent,
            "requestedDate": self.requested_date,
            "selectedSlot": self.selected_slot,
            "availableSlots": self.available_slots[:5],
            "patientName": self.patient_name,
            "phone": self.patient_phone,
            "awaitingConfirmation": self.is_awaiting_confirmation,
            "bookingConfirmed": self.booking_confirmed,
            "turns": self.turn_count,
            "elapsedSeconds": int(time.time() - self.start_timestamp),
        }

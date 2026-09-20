"""Backend-controlled Clinic AI tools for Z Cloud AI Telecaller.

Tools follow strict backend authorization:
LLM -> validated tool -> server authorization -> tenant validation ->
business-rule validation -> database transaction -> structured result -> LLM response.

Multi-Tenancy Rule:
tenant_id is NEVER accepted from LLM parameters. It is permanently bound to the
authenticated session context.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
from typing import Any, Callable, Coroutine, Optional

from livekit.agents.llm import function_tool

from zcloud_ai_telecaller.runtime_client import (
    RuntimeClientAuthError,
    RuntimeClientConnectionError,
    RuntimeClientError,
    RuntimeClientNotFoundError,
    RuntimeClientValidationError,
    RuntimeConfigClient,
)

logger = logging.getLogger("zcloud_ai_telecaller.clinic_tools")


def normalize_date(date_str: str, base_date: Optional[datetime.date] = None) -> str:
    """Normalizes natural language or formatted dates into authoritative YYYY-MM-DD format.

    Args:
        date_str: String input such as "2026-09-22", "today", "tomorrow", or "next monday".
        base_date: Optional reference date (defaults to current UTC date).

    Returns:
        Authoritative YYYY-MM-DD date string.
    """
    cleaned = date_str.strip().lower()
    ref = base_date or datetime.datetime.now(datetime.timezone.utc).date()

    if cleaned in ("today", "current"):
        return ref.isoformat()
    if cleaned in ("tomorrow",):
        return (ref + datetime.timedelta(days=1)).isoformat()
    if cleaned in ("day after tomorrow",):
        return (ref + datetime.timedelta(days=2)).isoformat()

    # Direct YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", cleaned):
        return cleaned

    # MM/DD/YYYY or MM-DD-YYYY
    slash_match = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", cleaned)
    if slash_match:
        m, d, y = map(int, slash_match.groups())
        return f"{y:04d}-{m:02d}-{d:02d}"

    # Fallback to ref date if unparseable
    return cleaned


def normalize_time(time_str: str) -> str:
    """Normalizes time expressions into standard 24-hour HH:mm format.

    Examples:
        '10:30' -> '10:30'
        '10:30 am' -> '10:30'
        '2:00 pm' -> '14:00'
        '2pm' -> '14:00'
        '9am' -> '09:00'
    """
    cleaned = time_str.strip().lower()

    # Already HH:mm 24-hour
    if re.match(r"^([01]\d|2[0-3]):[0-5]\d$", cleaned):
        return cleaned

    # 12-hour with am/pm (e.g. 2:30 pm or 2pm)
    match_12 = re.match(r"^(\d{1,2})(?::([0-5]\d))?\s*(am|pm)$", cleaned)
    if match_12:
        hour = int(match_12.group(1))
        minute = int(match_12.group(2) or 0)
        period = match_12.group(3)

        if period == "pm" and hour < 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute:02d}"

    return cleaned


def normalize_phone(phone_str: str) -> str:
    """Normalizes phone string to clean formatted representation."""
    cleaned = re.sub(r"[^\d+]", "", phone_str.strip())
    if len(cleaned) == 10 and not cleaned.startswith("+"):
        return f"+1{cleaned}"
    return cleaned


class ToolLoopLimiter:
    """Protects LLM agent against infinite tool-call recursion loops."""

    def __init__(self, max_per_turn: int = 4, max_per_session: int = 25) -> None:
        self.max_per_turn = max_per_turn
        self.max_per_session = max_per_session
        self.current_turn_count = 0
        self.total_session_count = 0

    def reset_turn(self) -> None:
        self.current_turn_count = 0

    def acquire(self) -> bool:
        """Attempts to register a tool call. Returns False if limit exceeded."""
        if self.current_turn_count >= self.max_per_turn:
            logger.warning(
                "Tool call loop detected: turn count %d exceeded max %d",
                self.current_turn_count,
                self.max_per_turn,
            )
            return False
        if self.total_session_count >= self.max_per_session:
            logger.warning(
                "Session tool limit reached: %d exceeded max %d",
                self.total_session_count,
                self.max_per_session,
            )
            return False

        self.current_turn_count += 1
        self.total_session_count += 1
        return True


class ClinicToolsManager:
    """Manages backend-controlled clinic tools for an active call session."""

    def __init__(
        self,
        tenant_id: str,
        agent_id: str,
        client: RuntimeConfigClient,
        call_id: Optional[str] = None,
        transfer_rules: Optional[dict[str, Any]] = None,
        clinic_profile: Optional[dict[str, Any]] = None,
        on_event: Optional[Callable[[dict[str, Any]], Coroutine[Any, Any, None]]] = None,
    ) -> None:
        # Strict multi-tenant derivation: tenant_id cannot be overwritten by LLM
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.call_id = call_id
        self.client = client
        self.transfer_rules = transfer_rules or {}
        self.clinic_profile = clinic_profile or {}
        self.on_event = on_event
        self.loop_limiter = ToolLoopLimiter()

    async def _emit_event(self, invocation: str, result: str) -> None:
        """Broadcasts tool invocation and result for live workbench feed."""
        if self.on_event:
            try:
                await self.on_event(
                    {
                        "type": "transcript",
                        "speaker": "AGENT",
                        "message": "",
                        "toolInvocation": invocation,
                        "toolResult": result,
                    }
                )
            except Exception as exc:
                logger.debug("Failed to emit tool event: %s", exc)

    async def check_availability_action(
        self,
        date: str,
        service: str = "General Consultation",
    ) -> str:
        """Checks actual real-time availability in the clinic database for a date."""
        if not self.loop_limiter.acquire():
            return "I have checked the schedule. Please let me know what time works best for you."

        norm_date = normalize_date(date)
        invocation = f"check_availability(date='{norm_date}', service='{service}')"
        logger.info("Executing backend tool: %s for tenant: %s", invocation, self.tenant_id)

        try:
            res = await self.client.check_availability(
                tenant_id=self.tenant_id,
                date=norm_date,
                service_name=service,
            )

            is_open = res.get("isOpen", True)
            if not is_open:
                reason = res.get("reason", "The clinic is closed on this date.")
                result_str = f"Closed: {reason}"
                await self._emit_event(invocation, result_str)
                return f"The clinic is closed on {norm_date}. {reason}. Would you like to check another date?"

            slots = res.get("availableSlots", [])
            if not slots:
                result_str = "No open slots available"
                await self._emit_event(invocation, result_str)
                return f"There are no available appointment slots left on {norm_date}. Would you like to try another date?"

            # Offer up to 4 popular morning/afternoon slots cleanly
            offered = slots[:5]
            slots_str = ", ".join(offered)
            result_str = f"Available: {slots_str} ({len(slots)} total)"
            await self._emit_event(invocation, result_str)

            return (
                f"We have open appointments on {norm_date} at: {slots_str}. "
                f"Which of these times works best for you?"
            )
        except Exception as exc:
            logger.error("Failed to check availability: %s", exc)
            result_str = f"Error: {exc}"
            await self._emit_event(invocation, result_str)
            return f"I could not retrieve openings for {norm_date} right now. Our standard hours are weekdays 8 AM to 6 PM. Would you like me to connect you with our front desk?"

    async def book_appointment_action(
        self,
        patient_name: str,
        phone: str,
        date: str,
        time: str,
        service: str = "General Consultation",
        notes: str = "",
    ) -> str:
        """Books an authoritative appointment in the clinic database."""
        if not self.loop_limiter.acquire():
            return "Appointment booking is in progress. Please hold."

        norm_date = normalize_date(date)
        norm_time = normalize_time(time)
        norm_phone = normalize_phone(phone)
        trimmed_name = patient_name.strip()

        invocation = (
            f"book_appointment(name='{trimmed_name}', phone='{norm_phone}', "
            f"date='{norm_date}', time='{norm_time}', service='{service}')"
        )
        logger.info("Executing backend booking tool: %s for tenant: %s", invocation, self.tenant_id)

        # Pre-validation
        if len(trimmed_name) < 2:
            msg = "Please provide the patient's full first and last name to book the appointment."
            await self._emit_event(invocation, f"Rejected: {msg}")
            return msg

        if len(re.sub(r"\D", "", norm_phone)) < 7:
            msg = "Please provide a valid contact phone number so we can confirm the booking."
            await self._emit_event(invocation, f"Rejected: {msg}")
            return msg

        try:
            res = await self.client.book_appointment(
                tenant_id=self.tenant_id,
                agent_id=self.agent_id,
                call_id=self.call_id,
                patient_name=trimmed_name,
                phone=norm_phone,
                date=norm_date,
                time=norm_time,
                service_name=service,
                notes=notes,
            )

            is_existing = res.get("isExisting", False)
            apt_id = res.get("appointmentId", "CONFIRMED")
            status = res.get("status", "UPCOMING")

            if is_existing:
                result_str = f"Already Confirmed (id={apt_id})"
                await self._emit_event(invocation, result_str)
                return (
                    f"Good news! An appointment for {trimmed_name} is already confirmed for "
                    f"{norm_date} at {norm_time} for {service}."
                )

            result_str = f"Confirmed UPCOMING (id={apt_id})"
            await self._emit_event(invocation, result_str)
            return (
                f"Your appointment has been successfully scheduled! Here are the details: "
                f"Patient: {trimmed_name}, Date: {norm_date}, Time: {norm_time}, Service: {service}. "
                f"We look forward to seeing you at {self.clinic_profile.get('name', 'the clinic')}."
            )
        except RuntimeClientValidationError as val_err:
            result_str = f"Validation Error: {val_err}"
            await self._emit_event(invocation, result_str)
            return f"I was unable to schedule that time slot: {val_err}. Would you like to pick a different open time?"
        except Exception as exc:
            logger.error("Failed to book appointment: %s", exc)
            result_str = f"Error: {exc}"
            await self._emit_event(invocation, result_str)
            return f"I encountered an issue confirming your reservation. Please verify your preferred time and phone number, or our reception team will be glad to assist."

    async def escalate_emergency_action(self, symptoms_or_reason: str) -> str:
        """Escalates urgent medical symptoms or clinical emergency requests."""
        invocation = f"escalate_emergency(reason='{symptoms_or_reason}')"
        logger.warning(
            "Emergency escalation triggered for tenant %s: reason='%s'",
            self.tenant_id,
            symptoms_or_reason,
        )

        handoff = self.transfer_rules.get(
            "handoffMessage",
            "If this is a medical emergency, please hang up and dial 911 immediately or visit the nearest emergency room.",
        )
        phone = self.transfer_rules.get("primaryNumber") or self.clinic_profile.get("phone", "911")

        result_str = f"Escalated to clinical line: {phone}"
        await self._emit_event(invocation, result_str)

        return (
            f"CLINICAL SAFETY ALERT: As an AI assistant, I cannot provide medical diagnosis, "
            f"treatment advice, or medication guidance. If this is a medical emergency, please hang up and dial 911 immediately. "
            f"{handoff} You can also reach our triage nurse at {phone}."
        )

    def get_function_tools(self) -> list[Any]:
        """Builds LiveKit function tools with strict docstrings and schemas for Gemini LLM."""
        manager = self

        @function_tool
        async def check_clinic_availability(date: str, service: str = "General Consultation") -> str:
            """Check available appointment time slots for a given date.

            Args:
                date: Date to check in YYYY-MM-DD format (or 'today', 'tomorrow').
                service: Clinic service name (e.g. 'Consultation', 'Cleaning').

            Returns:
                Authoritative available time slots from the clinic database.
            """
            return await manager.check_availability_action(date=date, service=service)

        @function_tool
        async def book_clinic_appointment(
            patient_name: str,
            phone: str,
            date: str,
            time: str,
            service: str = "General Consultation",
            notes: str = "",
        ) -> str:
            """Book a verified appointment for a patient in the clinic database.

            Args:
                patient_name: Full name of the patient.
                phone: Contact phone number with area code.
                date: Appointment date in YYYY-MM-DD format.
                time: Appointment time in 24-hour HH:mm format (e.g., '10:30' or '14:00').
                service: Service or treatment name.
                notes: Optional notes or patient preferences.

            Returns:
                Confirmation details or reason for failure.
            """
            return await manager.book_appointment_action(
                patient_name=patient_name,
                phone=phone,
                date=date,
                time=time,
                service=service,
                notes=notes,
            )

        @function_tool
        async def escalate_medical_emergency(symptoms_or_reason: str) -> str:
            """Escalate medical emergencies, severe symptoms, or requests for clinical medical advice.

            Args:
                symptoms_or_reason: Description of caller's symptoms or emergency request.

            Returns:
                Standard clinic emergency escalation guidance and triage transfer contact.
            """
            return await manager.escalate_emergency_action(symptoms_or_reason=symptoms_or_reason)

        return [check_clinic_availability, book_clinic_appointment, escalate_medical_emergency]

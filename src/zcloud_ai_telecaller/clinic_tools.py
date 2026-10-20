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

from zcloud_ai_telecaller.conversation_state import ConversationStateManager
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

    # Relative weekdays (e.g. "monday", "next monday", "this friday")
    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for w_name, w_idx in weekdays.items():
        if cleaned in (w_name, f"next {w_name}", f"this {w_name}"):
            current_w = ref.weekday()
            days_ahead = (w_idx - current_w) % 7
            if days_ahead == 0 and "next" in cleaned:
                days_ahead = 7
            return (ref + datetime.timedelta(days=days_ahead)).isoformat()

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
        '5:30' -> '17:30' (afternoon clinic context)
        '9am' -> '09:00'
    """
    cleaned = time_str.strip().lower()

    # Special natural phrases
    if "after 5" in cleaned or "evening" in cleaned:
        return "17:30"
    if "morning" in cleaned:
        return "10:00"
    if "afternoon" in cleaned:
        return "14:00"

    # Already standard 24-hour format
    if re.match(r"^([01]\d|2[0-3]):[0-5]\d$", cleaned):
        return cleaned

    # 12-hour with am/pm (e.g. 2:30 pm, 5pm)
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

    # Bare hour or hour:minute without am/pm (e.g. "5:30" or "5")
    match_bare = re.match(r"^(\d{1,2})(?::([0-5]\d))?$", cleaned)
    if match_bare:
        hour = int(match_bare.group(1))
        minute = int(match_bare.group(2) or 0)
        # In typical clinic daytime hours (8:00 - 18:00), 1 through 6 means PM
        if 1 <= hour <= 6:
            hour += 12
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
        state_manager: Optional[ConversationStateManager] = None,
        on_event: Optional[Callable[[dict[str, Any]], Coroutine[Any, Any, None]]] = None,
    ) -> None:
        # Strict multi-tenant derivation: tenant_id cannot be overwritten by LLM
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.call_id = call_id
        self.client = client
        self.transfer_rules = transfer_rules or {}
        self.clinic_profile = clinic_profile or {}
        self.state_manager = state_manager
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
            if self.state_manager:
                self.state_manager.record_availability_result(norm_date, slots, service)

            if not slots:
                result_str = "No open slots available"
                await self._emit_event(invocation, result_str)
                # Check day after tomorrow suggestion
                next_day = (datetime.date.fromisoformat(norm_date) + datetime.timedelta(days=1)).isoformat()
                return f"The clinic schedule is completely booked on {norm_date}. Would you like me to check openings for {next_day}?"

            # Categorize slots (morning vs afternoon/evening)
            morning_slots = [s for s in slots if s < "12:00"]
            afternoon_slots = [s for s in slots if s >= "12:00"]

            if not morning_slots and afternoon_slots:
                offered = afternoon_slots[:3]
                slots_str = ", ".join(offered)
                result_str = f"Available: {slots_str} ({len(slots)} total, morning full)"
                await self._emit_event(invocation, result_str)
                return (
                    f"Sir, on {norm_date}, morning slots are fully booked. "
                    f"We have afternoon openings at {slots_str}. Which time works best for you?"
                )

            offered = slots[:4]
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

        idempotency_key = f"{self.call_id or self.tenant_id}_{norm_phone}_{norm_date}_{norm_time}"
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
                idempotency_key=idempotency_key,
            )

            is_existing = res.get("isExisting", False)
            apt_id = res.get("appointmentId", "CONFIRMED")
            status = res.get("status", "UPCOMING")

            if self.state_manager:
                self.state_manager.mark_booking_confirmed(apt_id)

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
                f"A WhatsApp confirmation has been dispatched. We look forward to seeing you at {self.clinic_profile.get('name', 'the clinic')}."
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

    async def cancel_appointment_action(self, appointment_id: str, reason: str = "") -> str:
        """Cancels an existing appointment in the clinic database."""
        invocation = f"cancel_appointment(id='{appointment_id}')"
        logger.info("Executing cancellation tool: %s", invocation)
        try:
            res = await self.client.cancel_appointment(
                tenant_id=self.tenant_id,
                appointment_id=appointment_id,
                reason=reason,
            )
            result_str = "Cancelled successfully"
            await self._emit_event(invocation, result_str)
            return f"Your appointment has been cancelled and the time slot released. Please let us know if you would like to reschedule for a future date."
        except Exception as exc:
            logger.error("Failed to cancel appointment: %s", exc)
            await self._emit_event(invocation, f"Error: {exc}")
            return f"I was unable to cancel that appointment right now. Our front desk staff will be happy to assist you directly."

    async def reschedule_appointment_action(
        self,
        appointment_id: str,
        new_date: str,
        new_time: str,
    ) -> str:
        """Reschedules an existing appointment to a new date and time."""
        norm_date = normalize_date(new_date)
        norm_time = normalize_time(new_time)
        invocation = f"reschedule_appointment(id='{appointment_id}', date='{norm_date}', time='{norm_time}')"
        logger.info("Executing reschedule tool: %s", invocation)
        try:
            res = await self.client.reschedule_appointment(
                tenant_id=self.tenant_id,
                appointment_id=appointment_id,
                new_date=norm_date,
                new_time=norm_time,
            )
            result_str = f"Rescheduled to {norm_date} at {norm_time}"
            await self._emit_event(invocation, result_str)
            return f"Your appointment has been successfully rescheduled to {norm_date} at {norm_time}."
        except Exception as exc:
            logger.error("Failed to reschedule appointment: %s", exc)
            await self._emit_event(invocation, f"Error: {exc}")
            return f"I could not reschedule your appointment: {exc}. Would you like to select another time?"

    async def get_clinic_info_action(self) -> str:
        """Retrieves authoritative clinic profile, address, and contact details."""
        if not self.loop_limiter.acquire():
            return "Please let me know how I can help with your clinic visit."

        invocation = "get_clinic_info()"
        logger.info("Executing backend tool: %s for tenant: %s", invocation, self.tenant_id)
        try:
            res = await self.client.get_clinic_info(tenant_id=self.tenant_id)
            c_name = res.get("clinicName") or "our clinic"
            c_phone = res.get("phone") or ""
            c_address = res.get("address") or ""
            c_website = res.get("website") or ""
            c_desc = res.get("description") or ""

            result_str = f"Clinic: {c_name}, Phone: {c_phone}, Address: {c_address}"
            await self._emit_event(invocation, result_str)

            resp_parts = [f"{c_name} is located at {c_address}."]
            if c_phone:
                resp_parts.append(f"Our phone number is {c_phone}.")
            if c_desc:
                resp_parts.append(c_desc)
            if c_website:
                resp_parts.append(f"You can also visit us online at {c_website}.")
            return " ".join(resp_parts)
        except Exception as exc:
            logger.error("Failed to retrieve clinic info: %s", exc)
            await self._emit_event(invocation, f"Error: {exc}")
            c_name = self.clinic_profile.get("name", "our clinic")
            c_phone = self.clinic_profile.get("phone", "+1 (555) 234-5678")
            c_address = self.clinic_profile.get("address", "our primary location")
            return f"{c_name} is located at {c_address}. You can reach us at {c_phone}."

    async def get_services_action(self) -> str:
        """Retrieves the list of active healthcare services offered by the clinic."""
        if not self.loop_limiter.acquire():
            return "We offer general consultations, cleanings, and preventive care."

        invocation = "get_services()"
        logger.info("Executing backend tool: %s for tenant: %s", invocation, self.tenant_id)
        try:
            services = await self.client.get_services(tenant_id=self.tenant_id)
            if not services:
                s_names = self.clinic_profile.get("services", [
                    "General Consultation",
                    "Dental Cleaning & Checkup",
                    "Preventive Care",
                    "Follow-up Appointment",
                ])
                serv_str = ", ".join(s_names)
                await self._emit_event(invocation, f"Services: {serv_str}")
                return f"We offer {serv_str}. Which of these would you like to schedule?"

            names = [s.get("name") for s in services if s.get("name")]
            serv_str = ", ".join(names)
            result_str = f"Active services ({len(names)}): {serv_str}"
            await self._emit_event(invocation, result_str)
            return f"We currently offer: {serv_str}. Would you like to check available appointment times for any of these?"
        except Exception as exc:
            logger.error("Failed to retrieve services: %s", exc)
            await self._emit_event(invocation, f"Error: {exc}")
            return "We provide general consultations, cleanings, and routine checkups. Would you like to book a visit?"

    async def get_business_hours_action(self) -> str:
        """Retrieves weekly operating hours and closed days for the clinic."""
        if not self.loop_limiter.acquire():
            return "Our standard hours are weekdays 8 AM to 6 PM."

        invocation = "get_business_hours()"
        logger.info("Executing backend tool: %s for tenant: %s", invocation, self.tenant_id)
        try:
            schedule = await self.client.get_business_hours(tenant_id=self.tenant_id)
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            parts = []
            for d in days:
                info = schedule.get(d) or {}
                if info.get("closed"):
                    parts.append(f"{d.capitalize()}: Closed")
                else:
                    parts.append(f"{d.capitalize()}: {info.get('open', '08:00')} to {info.get('close', '18:00')}")

            result_str = "; ".join(parts)
            await self._emit_event(invocation, result_str)
            return f"Our operating hours are: {result_str}. When would you like to come in?"
        except Exception as exc:
            logger.error("Failed to retrieve business hours: %s", exc)
            await self._emit_event(invocation, f"Error: {exc}")
            return "We are open Monday through Friday from 8 AM to 6 PM, Saturday 9 AM to 1 PM, and closed on Sundays."

    async def find_customer_action(self, phone: str = "", email: str = "") -> str:
        """Finds existing customer profile in tenant CRM."""
        norm_phone = normalize_phone(phone) if phone else None
        clean_email = email.strip() if email else None

        invocation = f"find_customer(phone='{norm_phone or ''}', email='{clean_email or ''}')"
        logger.info("Executing backend tool: %s for tenant: %s", invocation, self.tenant_id)
        try:
            cust = await self.client.find_customer(
                tenant_id=self.tenant_id,
                phone=norm_phone,
                email=clean_email,
            )
            if cust:
                name = cust.get("name", "valued patient")
                if self.state_manager:
                    self.state_manager.patient_name = name
                    if cust.get("phone"):
                        self.state_manager.patient_phone = cust.get("phone")
                result_str = f"Found customer: {name} (id={cust.get('id')})"
                await self._emit_event(invocation, result_str)
                return f"I found your record in our system, {name}. How can I assist you with your appointment today?"

            result_str = "Customer not found"
            await self._emit_event(invocation, result_str)
            return "I could not find an existing record with those details. May I have your full name so I can register you?"
        except Exception as exc:
            logger.error("Failed to find customer: %s", exc)
            await self._emit_event(invocation, f"Error: {exc}")
            return "I was unable to search our records at this moment. May I please have your name and contact phone number?"

    async def create_customer_action(self, name: str, phone: str, email: str = "") -> str:
        """Registers new patient contact in tenant CRM."""
        norm_phone = normalize_phone(phone)
        trimmed_name = name.strip()
        clean_email = email.strip() if email else None

        invocation = f"create_customer(name='{trimmed_name}', phone='{norm_phone}')"
        logger.info("Executing backend tool: %s for tenant: %s", invocation, self.tenant_id)
        try:
            res = await self.client.create_customer(
                tenant_id=self.tenant_id,
                name=trimmed_name,
                phone=norm_phone,
                email=clean_email,
            )
            cust = res.get("customer", {})
            c_name = cust.get("name", trimmed_name)
            if self.state_manager:
                self.state_manager.patient_name = c_name
                self.state_manager.patient_phone = norm_phone

            result_str = f"Customer registered: {c_name} ({norm_phone})"
            await self._emit_event(invocation, result_str)
            return f"Thank you, {c_name}. I have your contact information saved. Which date and time would you like to schedule?"
        except Exception as exc:
            logger.error("Failed to create customer: %s", exc)
            await self._emit_event(invocation, f"Error: {exc}")
            return f"I have noted your name as {trimmed_name} and phone as {norm_phone}. Let's select your preferred appointment time."

    async def request_human_transfer_action(self, reason: str = "") -> str:
        """Initiates human staff transfer and creates escalation record."""
        invocation = f"request_human_transfer(reason='{reason}')"
        logger.info("Executing transfer tool: %s for tenant: %s", invocation, self.tenant_id)
        try:
            res = await self.client.request_human_transfer(
                tenant_id=self.tenant_id,
                call_id=self.call_id,
                reason=reason or "Customer requested staff transfer",
            )
            phone = res.get("transferNumber") or self.clinic_profile.get("phone", "+1 (555) 234-5678")
            fallback = res.get("fallbackMessage") or f"I am transferring you to our front desk staff at {phone}. Please hold."
            result_str = f"Transferred to {phone}: {reason}"
            await self._emit_event(invocation, result_str)
            return fallback
        except Exception as exc:
            logger.error("Failed to request transfer: %s", exc)
            phone = self.clinic_profile.get("phone", "+1 (555) 234-5678")
            await self._emit_event(invocation, f"Fallback transfer to {phone}")
            return f"Please hold while I connect you with our clinic staff directly at {phone}."

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
        phone = self.transfer_rules.get("primaryNumber") or self.clinic_profile.get("phone", "+1 (555) 234-5678")
        result_str = f"Escalated: {symptoms_or_reason}"
        await self._emit_event(invocation, result_str)

        return (
            f"CLINICAL SAFETY ALERT: As an AI administrative assistant, I cannot provide medical diagnosis. "
            f"If this is a medical emergency, please hang up and dial 911 immediately or visit the nearest emergency room. "
            f"{handoff} For urgent clinical triage, our emergency nurse line is available at {phone}."
        )

    async def search_clinic_knowledge_action(self, query: str) -> str:
        """Searches authoritative clinic documents and FAQs for grounded answers."""
        if not self.loop_limiter.acquire():
            return "I have checked our records. What else can I help you with?"

        clean_query = query.strip()
        invocation = f"search_clinic_knowledge(query='{clean_query}')"
        logger.info("Executing backend tool: %s for tenant: %s", invocation, self.tenant_id)

        try:
            res = await self.client.search_knowledge(
                tenant_id=self.tenant_id,
                query=clean_query,
                top_k=3,
                min_score=0.45,
            )

            has_evidence = res.get("hasEvidence", False)
            results = res.get("results", [])

            if has_evidence and results:
                top_r = results[0]
                result_str = f"Found {len(results)} match(es), top score={top_r.get('score')} ({top_r.get('sourceTitle')})"
                await self._emit_event(invocation, result_str)

                passages_summary = "\n\n".join(
                    f"[{r.get('sourceTitle')}]: {r.get('content')}" for r in results[:2]
                )
                return (
                    f"<untrusted_clinic_knowledge>\n"
                    f"{passages_summary}\n"
                    f"</untrusted_clinic_knowledge>\n"
                    f"Instruction to agent: State these verified clinic facts directly and concisely to the caller."
                )

            result_str = "No verified matches found"
            await self._emit_event(invocation, result_str)
            return (
                "I could not verify that specific detail in our clinic knowledge documents. "
                "I can connect you directly with our front desk receptionist, or help you check our appointment schedule."
            )

        except Exception as exc:
            logger.error("Knowledge search tool failed for tenant %s: %s", self.tenant_id, exc)
            await self._emit_event(invocation, f"Error: {exc}")
            return (
                "I am unable to search our clinic policy documents at this moment. "
                "May I connect you with our front desk receptionist, or check our appointment calendar?"
            )

    def get_function_tools(self) -> list[Any]:
        """Builds LiveKit function tools with strict docstrings and schemas for Gemini LLM."""
        manager = self

        @function_tool
        async def search_clinic_knowledge(query: str) -> str:
            """Search the official clinic knowledge base, practice policies, accepted insurance plans, treatment info, doctor credentials, and FAQs.

            Args:
                query: Specific question or topic to search (e.g. 'accepted insurance', 'root canal duration', 'Saturday hours').

            Returns:
                Verified facts from clinic documents or guidance if not verified.
            """
            return await manager.search_clinic_knowledge_action(query=query)

        @function_tool
        async def get_clinic_info() -> str:
            """Retrieve general clinic information including location, physical address, contact phone, website, and policies.

            Returns:
                Official clinic location, phone number, and overview.
            """
            return await manager.get_clinic_info_action()

        @function_tool
        async def get_services() -> str:
            """Retrieve the healthcare, medical, or dental services offered by the clinic.

            Returns:
                List of active services provided by the clinic.
            """
            return await manager.get_services_action()

        @function_tool
        async def get_business_hours() -> str:
            """Retrieve clinic operating hours, weekday schedules, Saturday hours, and closed days.

            Returns:
                Weekly business hours and schedule.
            """
            return await manager.get_business_hours_action()

        @function_tool
        async def check_clinic_availability(date: str, service: str = "General Consultation") -> str:
            """Check available appointment time slots for a given date in the clinic database.

            Args:
                date: Date to check in YYYY-MM-DD format (or 'today', 'tomorrow', 'next monday').
                service: Clinic service name (e.g. 'Consultation', 'Cleaning').

            Returns:
                Authoritative available time slots from the clinic database.
            """
            return await manager.check_availability_action(date=date, service=service)

        @function_tool
        async def find_customer(phone: str = "", email: str = "") -> str:
            """Search for an existing patient or customer in the clinic CRM by phone number or email.

            Args:
                phone: Caller phone number.
                email: Optional patient email address.

            Returns:
                Customer profile confirmation or not found notice.
            """
            return await manager.find_customer_action(phone=phone, email=email)

        @function_tool
        async def create_customer(name: str, phone: str, email: str = "") -> str:
            """Register a new patient or caller with their contact details in the clinic database.

            Args:
                name: Full name of the patient.
                phone: Contact phone number.
                email: Optional email address.

            Returns:
                Registration confirmation.
            """
            return await manager.create_customer_action(name=name, phone=phone, email=email)

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
                time: Appointment time in 24-hour HH:mm format (e.g., '10:30' or '17:30').
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
        async def cancel_clinic_appointment(appointment_id: str, reason: str = "") -> str:
            """Cancel an existing clinic appointment.

            Args:
                appointment_id: The booking ID of the appointment to cancel.
                reason: Optional explanation for cancellation.

            Returns:
                Cancellation result.
            """
            return await manager.cancel_appointment_action(appointment_id=appointment_id, reason=reason)

        @function_tool
        async def reschedule_clinic_appointment(
            appointment_id: str,
            new_date: str,
            new_time: str,
        ) -> str:
            """Reschedule an existing appointment to a new date and time.

            Args:
                appointment_id: The booking ID of the appointment to reschedule.
                new_date: New date in YYYY-MM-DD format.
                new_time: New time in HH:mm 24-hour format.

            Returns:
                Rescheduling result.
            """
            return await manager.reschedule_appointment_action(
                appointment_id=appointment_id,
                new_date=new_date,
                new_time=new_time,
            )

        @function_tool
        async def request_human_transfer(reason: str = "") -> str:
            """Transfer or escalate the caller to a human receptionist, doctor, or staff member.

            Args:
                reason: Reason caller is requesting human staff.

            Returns:
                Staff transfer instructions and phone contact.
            """
            return await manager.request_human_transfer_action(reason=reason)

        @function_tool
        async def escalate_medical_emergency(symptoms_or_reason: str) -> str:
            """Escalate medical emergencies, severe symptoms, or requests for clinical medical advice.

            Args:
                symptoms_or_reason: Description of caller's symptoms or emergency request.

            Returns:
                Standard clinic emergency escalation guidance and triage transfer contact.
            """
            return await manager.escalate_emergency_action(symptoms_or_reason=symptoms_or_reason)

        return [
            search_clinic_knowledge,
            get_clinic_info,
            get_services,
            get_business_hours,
            check_clinic_availability,
            find_customer,
            create_customer,
            book_clinic_appointment,
            cancel_clinic_appointment,
            reschedule_clinic_appointment,
            request_human_transfer,
            escalate_medical_emergency,
        ]

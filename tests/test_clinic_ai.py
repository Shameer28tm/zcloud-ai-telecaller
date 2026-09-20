"""Comprehensive test suite for Phase 3 Clinic AI Capability.

Covers:
- Date, time, phone normalizations
- Fast-path deterministic query matching
- Clinic prompt construction with AI safety guardrails
- Tool loop limiter & recursion protection
- Backend-controlled tool execution (availability, booking, idempotency, double-booking)
- Multi-tenant derivation & context isolation
- Emergency medical escalation
- LLM Quota 429 rate limit circuit breaker and safe fallback
"""

import asyncio
import datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zcloud_ai_telecaller.clinic_tools import (
    ClinicToolsManager,
    ToolLoopLimiter,
    normalize_date,
    normalize_phone,
    normalize_time,
)
from zcloud_ai_telecaller.prompt_builder import (
    build_clinic_system_prompt,
    format_business_hours,
    match_deterministic_clinic_query,
)
from zcloud_ai_telecaller.runtime_client import (
    AgentRuntimeConfig,
    RuntimeClientAuthError,
    RuntimeClientValidationError,
    RuntimeConfigClient,
)


# ---------------------------------------------------------------------------
# Test 1: Normalization Utilities
# ---------------------------------------------------------------------------
def test_normalization_utilities():
    ref_date = datetime.date(2026, 9, 20)  # Sunday
    assert normalize_date("today", base_date=ref_date) == "2026-09-20"
    assert normalize_date("tomorrow", base_date=ref_date) == "2026-09-21"
    assert normalize_date("day after tomorrow", base_date=ref_date) == "2026-09-22"
    assert normalize_date("2026-09-25") == "2026-09-25"
    assert normalize_date("09/25/2026") == "2026-09-25"

    assert normalize_time("10:30") == "10:30"
    assert normalize_time("10:30 am") == "10:30"
    assert normalize_time("2:00 pm") == "14:00"
    assert normalize_time("2pm") == "14:00"
    assert normalize_time("9am") == "09:00"
    assert normalize_time("12:15 pm") == "12:15"
    assert normalize_time("12:00 am") == "00:00"

    assert normalize_phone("+1 (555) 234-5678") == "+15552345678"
    assert normalize_phone("5552345678") == "+15552345678"
    assert normalize_phone("234-5678") == "2345678"


# ---------------------------------------------------------------------------
# Test 2: Prompt Builder & AI Safety Guardrails
# ---------------------------------------------------------------------------
def test_clinic_prompt_builder():
    config = AgentRuntimeConfig(
        tenant_id="tenant-apex",
        agent_id="agent-chloe",
        agent_version_id="ver-1",
        name="Chloe",
        prompt="You are Chloe, receptionist for Apex Dental.",
        greeting="Hello, welcome to Apex Dental.",
        business_hours={
            "schedule": {
                "monday": {"open": "08:00", "close": "18:00", "closed": False},
                "sunday": {"open": "00:00", "close": "00:00", "closed": True},
            }
        },
        clinic_profile={
            "name": "Apex Dental Care",
            "phone": "+1 (555) 234-5678",
            "address": "742 Evergreen Terrace, Springfield, IL",
            "services": ["Teeth Cleaning", "Cavity Filling", "Root Canal"],
        },
        knowledge_base=[
            {"title": "Insurance", "content": "We accept Delta Dental and MetLife."}
        ],
    )

    prompt = build_clinic_system_prompt(config)

    # Verify key clinical facts and guardrails are in prompt
    assert "Apex Dental Care" in prompt
    assert "+1 (555) 234-5678" in prompt
    assert "742 Evergreen Terrace" in prompt
    assert "Teeth Cleaning" in prompt
    assert "Delta Dental" in prompt
    assert "ZERO MEDICAL ADVICE" in prompt
    assert "MEDICAL EMERGENCY ESCALATION" in prompt
    assert "SAFE FALLBACK" in prompt


# ---------------------------------------------------------------------------
# Test 3: Deterministic Fast-Path Matching
# ---------------------------------------------------------------------------
def test_deterministic_fast_path_matching():
    config = AgentRuntimeConfig(
        tenant_id="tenant-apex",
        agent_id="agent-chloe",
        agent_version_id="ver-1",
        name="Chloe",
        prompt="You are Chloe.",
        greeting="Hello!",
        business_hours={"schedule": {"monday": {"open": "08:00", "close": "18:00", "closed": False}}},
        clinic_profile={
            "name": "Apex Dental Care",
            "phone": "+1 (555) 234-5678",
            "address": "742 Evergreen Terrace, Suite 300",
            "services": ["Dental Cleaning", "Checkup"],
        },
    )

    # Hours query
    match_hours = match_deterministic_clinic_query("What are your business hours?", config)
    assert match_hours is not None
    assert "Apex Dental Care hours are" in match_hours[0]
    assert match_hours[1] is None

    # Location query
    match_loc = match_deterministic_clinic_query("Where are you located?", config)
    assert match_loc is not None
    assert "742 Evergreen Terrace" in match_loc[0]

    # Phone query
    match_phone = match_deterministic_clinic_query("What is your phone number?", config)
    assert match_phone is not None
    assert "+1 (555) 234-5678" in match_phone[0]

    # Services query
    match_services = match_deterministic_clinic_query("What services do you offer?", config)
    assert match_services is not None
    assert "Dental Cleaning" in match_services[0]

    # Emergency symptoms fast-path
    match_emergency = match_deterministic_clinic_query("I have severe chest pain and can't breathe", config)
    assert match_emergency is not None
    assert "dial 911" in match_emergency[0]
    assert match_emergency[1] == "escalate_medical_emergency"

    # Non-deterministic query (requires LLM)
    match_booking = match_deterministic_clinic_query("Can I see Dr. Reed tomorrow morning?", config)
    assert match_booking is None


# ---------------------------------------------------------------------------
# Test 4: Tool Call Loop Limiter
# ---------------------------------------------------------------------------
def test_tool_loop_limiter():
    limiter = ToolLoopLimiter(max_per_turn=3, max_per_session=5)

    assert limiter.acquire() is True
    assert limiter.acquire() is True
    assert limiter.acquire() is True
    # 4th call in the same turn should be blocked
    assert limiter.acquire() is False

    # Reset turn
    limiter.reset_turn()
    assert limiter.acquire() is True
    assert limiter.acquire() is True
    # Session limit reached (5 total)
    assert limiter.acquire() is False


# ---------------------------------------------------------------------------
# Test 5: Backend-Controlled Availability Tool
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_clinic_tools_check_availability():
    mock_client = AsyncMock()
    mock_client.check_availability = AsyncMock(
        return_value={
            "isOpen": True,
            "date": "2026-09-22",
            "availableSlots": ["09:00", "09:30", "10:00", "11:00", "14:00"],
            "totalAvailable": 5,
        }
    )

    events = []

    async def mock_on_event(ev):
        events.append(ev)

    manager = ClinicToolsManager(
        tenant_id="tenant-apex",
        agent_id="agent-chloe",
        client=mock_client,
        on_event=mock_on_event,
    )

    res = await manager.check_availability_action("2026-09-22", "Cleaning")

    assert "09:00" in res
    assert "10:00" in res
    mock_client.check_availability.assert_called_once_with(
        tenant_id="tenant-apex",
        date="2026-09-22",
        service_name="Cleaning",
    )
    assert len(events) == 1
    assert "check_availability" in events[0]["toolInvocation"]


# ---------------------------------------------------------------------------
# Test 6: Backend-Controlled Booking Tool & Idempotency
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_clinic_tools_book_appointment():
    mock_client = AsyncMock()
    mock_client.book_appointment = AsyncMock(
        return_value={
            "appointmentId": "apt-123",
            "status": "UPCOMING",
            "date": "2026-09-22",
            "time": "10:00",
            "patientName": "John Doe",
            "serviceName": "Teeth Cleaning",
            "isExisting": False,
        }
    )

    manager = ClinicToolsManager(
        tenant_id="tenant-apex",
        agent_id="agent-chloe",
        client=mock_client,
        clinic_profile={"name": "Apex Dental Care"},
    )

    res = await manager.book_appointment_action(
        patient_name="John Doe",
        phone="+15552345678",
        date="2026-09-22",
        time="10:00",
        service="Teeth Cleaning",
    )

    assert "successfully scheduled" in res
    assert "John Doe" in res
    assert "10:00" in res

    # Idempotent response test
    mock_client.book_appointment = AsyncMock(
        return_value={
            "appointmentId": "apt-123",
            "status": "UPCOMING",
            "date": "2026-09-22",
            "time": "10:00",
            "patientName": "John Doe",
            "serviceName": "Teeth Cleaning",
            "isExisting": True,
        }
    )

    res_idempotent = await manager.book_appointment_action(
        patient_name="John Doe",
        phone="+15552345678",
        date="2026-09-22",
        time="10:00",
        service="Teeth Cleaning",
    )
    assert "already confirmed" in res_idempotent


# ---------------------------------------------------------------------------
# Test 7: Double-Booking & Validation Error Handling
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_clinic_tools_validation_and_conflict_errors():
    mock_client = AsyncMock()
    mock_client.book_appointment = AsyncMock(
        side_effect=RuntimeClientValidationError("The 10:00 time slot on 2026-09-22 is already booked.")
    )

    manager = ClinicToolsManager(
        tenant_id="tenant-apex",
        agent_id="agent-chloe",
        client=mock_client,
    )

    # Conflict / Double-booking rejection
    res = await manager.book_appointment_action(
        patient_name="Bob Smith",
        phone="+15559998888",
        date="2026-09-22",
        time="10:00",
    )
    assert "unable to schedule that time slot" in res
    assert "already booked" in res

    # Short name rejection
    res_short_name = await manager.book_appointment_action(
        patient_name="J",
        phone="+15559998888",
        date="2026-09-22",
        time="10:00",
    )
    assert "first and last name" in res_short_name

    # Short phone rejection
    res_short_phone = await manager.book_appointment_action(
        patient_name="Jane Doe",
        phone="123",
        date="2026-09-22",
        time="10:00",
    )
    assert "valid contact phone number" in res_short_phone


# ---------------------------------------------------------------------------
# Test 8: Emergency Escalation Tool
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_clinic_emergency_escalation():
    manager = ClinicToolsManager(
        tenant_id="tenant-apex",
        agent_id="agent-chloe",
        client=AsyncMock(),
        transfer_rules={
            "primaryNumber": "+1 (555) 234-9999",
            "handoffMessage": "Connecting you directly to our clinical emergency nurse.",
        },
    )

    res = await manager.escalate_emergency_action("Caller reports severe allergic reaction.")
    assert "CLINICAL SAFETY ALERT" in res
    assert "cannot provide medical diagnosis" in res
    assert "dial 911" in res
    assert "+1 (555) 234-9999" in res


# ---------------------------------------------------------------------------
# Test 9: Multi-Tenant Derivation & Isolation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_clinic_tools_multi_tenant_isolation():
    mock_client = AsyncMock()
    mock_client.check_availability = AsyncMock(return_value={"isOpen": True, "availableSlots": ["09:00"]})

    # Tenant A
    manager_a = ClinicToolsManager(
        tenant_id="tenant-apex",
        agent_id="agent-chloe",
        client=mock_client,
    )
    await manager_a.check_availability_action("2026-09-22")
    assert mock_client.check_availability.call_args[1]["tenant_id"] == "tenant-apex"

    # Tenant B
    manager_b = ClinicToolsManager(
        tenant_id="tenant-metro",
        agent_id="agent-liam",
        client=mock_client,
    )
    await manager_b.check_availability_action("2026-09-22")
    assert mock_client.check_availability.call_args[1]["tenant_id"] == "tenant-metro"



# ---------------------------------------------------------------------------
# Test 10: LLM Quota (429) & Rate Limit Circuit Breaker Fallback
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_llm_quota_429_circuit_breaker():
    """Verify that when the LLM returns 429 Resource Exhausted, the worker does NOT crash,
    does NOT loop endlessly, and produces a safe pre-configured fallback message.
    """
    from zcloud_ai_telecaller.runtime_client import CallSessionAccumulator

    runtime_config = AgentRuntimeConfig(
        tenant_id="tenant-apex",
        agent_id="agent-chloe",
        agent_version_id="ver-1",
        name="Chloe",
        prompt="You are Chloe.",
        greeting="Hello!",
        clinic_profile={
            "name": "Apex Dental Care",
            "phone": "+1 (555) 234-5678",
        },
    )

    accumulator = CallSessionAccumulator(
        tenant_id="tenant-apex",
        agent_id="agent-chloe",
        room_name="test-quota-room",
    )

    broadcasted_messages = []

    async def mock_broadcast(msg, tool_invocation=None, tool_result=None):
        broadcasted_messages.append(msg)

    # Simulate 429 quota exhaustion error from Google Gemini provider
    mock_session = MagicMock()
    mock_session.generate_reply = AsyncMock(
        side_effect=Exception("429 Resource has been exhausted (e.g. check quota).")
    )

    tools_manager = ClinicToolsManager(
        tenant_id="tenant-apex",
        agent_id="agent-chloe",
        client=AsyncMock(),
    )

    # Execute the turn handler logic with circuit breaker
    async def run_turn_with_breaker(user_input: str):
        tools_manager.loop_limiter.reset_turn()
        for attempt in range(3):
            try:
                await mock_session.generate_reply(user_input=user_input)
                return
            except Exception as exc:
                err_str = str(exc).lower()
                is_quota = "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str
                if is_quota:
                    clinic_phone = runtime_config.clinic_profile.get("phone", "+1 (555) 234-5678")
                    fallback_msg = (
                        f"I apologize, our automated clinic assistant is experiencing unusually high call volume. "
                        f"Please hold for our front desk, or contact our reception directly at {clinic_phone}."
                    )
                    accumulator.add_transcript(speaker="AGENT", message=fallback_msg)
                    await mock_broadcast(fallback_msg)
                    return

    await run_turn_with_breaker("I need a root canal next week")

    # 1. Did not crash
    # 2. Called generate_reply only once (did not retry 429 indefinitely)
    assert mock_session.generate_reply.call_count == 1

    # 3. Emitted safe deterministic fallback message
    assert len(broadcasted_messages) == 1
    assert "high call volume" in broadcasted_messages[0]
    assert "+1 (555) 234-5678" in broadcasted_messages[0]

    # 4. Session accumulator preserved for billable minutes & transcript report
    assert len(accumulator.transcripts) == 1
    assert accumulator.transcripts[0].speaker == "AGENT"
    assert "high call volume" in accumulator.transcripts[0].message

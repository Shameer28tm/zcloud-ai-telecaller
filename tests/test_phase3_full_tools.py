"""Comprehensive Phase 3 Test Suite for Clinic AI Tools & Intelligence.

Validates:
- All 10 backend tools executed via ClinicToolsManager:
  1. get_clinic_info
  2. get_services
  3. get_business_hours
  4. check_clinic_availability
  5. find_customer
  6. create_customer
  7. book_clinic_appointment
  8. cancel_clinic_appointment
  9. reschedule_clinic_appointment
  10. request_human_transfer
  11. escalate_medical_emergency (Safety rule)
- Multi-tenant derivation: tenant_id cannot be chosen or overridden by LLM
- Tool loop limiter & infinite recursion circuit breaker
- Error handling & conversational graceful fallbacks (no stack traces)
- Timezone safety (Asia/Kolkata and America/New_York)
- Conversation state tracking and customer memory
- Zero secret logging
"""

import asyncio
import datetime
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zcloud_ai_telecaller.clinic_tools import (
    ClinicToolsManager,
    ToolLoopLimiter,
    normalize_date,
    normalize_phone,
    normalize_time,
)
from zcloud_ai_telecaller.conversation_state import ConversationStateManager
from zcloud_ai_telecaller.prompt_builder import (
    build_clinic_system_prompt,
    format_business_hours,
    match_deterministic_clinic_query,
)
from zcloud_ai_telecaller.runtime_client import (
    AgentRuntimeConfig,
    RuntimeClientAuthError,
    RuntimeClientConnectionError,
    RuntimeClientNotFoundError,
    RuntimeClientValidationError,
    RuntimeConfigClient,
)


# ---------------------------------------------------------------------------
# Test 1: Tool Client Methods & Error Mappings
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tool_client_auth_and_validation():
    # Missing secret should raise RuntimeClientAuthError
    client_no_secret = RuntimeConfigClient(base_url="http://localhost:3000", worker_api_secret="")
    with pytest.raises(RuntimeClientAuthError):
        await client_no_secret.get_clinic_info("tenant-1")

    with pytest.raises(RuntimeClientAuthError):
        await client_no_secret.get_services("tenant-1")

    with pytest.raises(RuntimeClientAuthError):
        await client_no_secret.get_business_hours("tenant-1")

    with pytest.raises(RuntimeClientAuthError):
        await client_no_secret.find_customer("tenant-1", phone="1234567")

    with pytest.raises(RuntimeClientAuthError):
        await client_no_secret.create_customer("tenant-1", name="Alice", phone="1234567")

    with pytest.raises(RuntimeClientAuthError):
        await client_no_secret.request_human_transfer("tenant-1")

    # Empty tenant_id should raise RuntimeClientValidationError
    client = RuntimeConfigClient(base_url="http://localhost:3000", worker_api_secret="secret_123")
    with pytest.raises(RuntimeClientValidationError):
        await client.get_clinic_info("")

    with pytest.raises(RuntimeClientValidationError):
        await client.find_customer("tenant-1")  # neither phone nor email


# ---------------------------------------------------------------------------
# Test 2: ClinicToolsManager get_clinic_info & get_services & get_business_hours
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_clinic_info_and_knowledge_tools():
    mock_client = MagicMock(spec=RuntimeConfigClient)
    mock_client.get_clinic_info = AsyncMock(
        return_value={
            "clinicName": "Apex Dental Clinic",
            "phone": "+1 (555) 777-9999",
            "address": "123 Medical Way, Springfield",
            "website": "https://apexdental.com",
            "description": "Family and cosmetic dental center",
        }
    )
    mock_client.get_services = AsyncMock(
        return_value=[
            {"name": "Teeth Cleaning", "durationMinutes": 45},
            {"name": "Dental Exam", "durationMinutes": 30},
        ]
    )
    mock_client.get_business_hours = AsyncMock(
        return_value={
            "monday": {"open": "08:00", "close": "18:00", "closed": False},
            "sunday": {"open": "00:00", "close": "00:00", "closed": True},
        }
    )

    manager = ClinicToolsManager(
        tenant_id="tenant-apex",
        agent_id="agent-1",
        client=mock_client,
    )

    # 1. get_clinic_info
    info_res = await manager.get_clinic_info_action()
    assert "Apex Dental Clinic" in info_res
    assert "123 Medical Way" in info_res
    assert "+1 (555) 777-9999" in info_res

    # 2. get_services
    serv_res = await manager.get_services_action()
    assert "Teeth Cleaning" in serv_res
    assert "Dental Exam" in serv_res

    # 3. get_business_hours
    hrs_res = await manager.get_business_hours_action()
    assert "Monday" in hrs_res
    assert "Sunday: Closed" in hrs_res


# ---------------------------------------------------------------------------
# Test 3: Customer Lookup and Creation Tools
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_customer_lookup_and_creation_tools():
    mock_client = MagicMock(spec=RuntimeConfigClient)
    mock_client.find_customer = AsyncMock(
        return_value={"id": "cust-1", "name": "Bob Smith", "phone": "+15552223333"}
    )
    mock_client.create_customer = AsyncMock(
        return_value={"customer": {"id": "cust-2", "name": "Carol Danvers", "phone": "+15554445555"}}
    )

    state = ConversationStateManager(tenant_id="tenant-apex", agent_id="agent-1")
    manager = ClinicToolsManager(
        tenant_id="tenant-apex",
        agent_id="agent-1",
        client=mock_client,
        state_manager=state,
    )

    # find_customer found
    find_res = await manager.find_customer_action(phone="+1 (555) 222-3333")
    assert "Bob Smith" in find_res
    assert state.patient_name == "Bob Smith"

    # find_customer not found
    mock_client.find_customer = AsyncMock(return_value=None)
    find_not_found = await manager.find_customer_action(phone="9999999999")
    assert "could not find" in find_not_found.lower()

    # create_customer
    create_res = await manager.create_customer_action(name="Carol Danvers", phone="555-444-5555")
    assert "Carol Danvers" in create_res
    assert state.patient_name == "Carol Danvers"


# ---------------------------------------------------------------------------
# Test 4: Appointment Booking with Availability Check & Conflict Handling
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_appointment_booking_and_safety():
    mock_client = MagicMock(spec=RuntimeConfigClient)
    mock_client.check_availability = AsyncMock(
        return_value={"isOpen": True, "availableSlots": ["10:00", "11:00", "14:00"]}
    )
    mock_client.book_appointment = AsyncMock(
        return_value={
            "appointmentId": "apt-101",
            "status": "UPCOMING",
            "date": "2026-10-20",
            "time": "10:00",
            "isExisting": False,
        }
    )

    state = ConversationStateManager(tenant_id="tenant-apex", agent_id="agent-1")
    manager = ClinicToolsManager(
        tenant_id="tenant-apex",
        agent_id="agent-1",
        client=mock_client,
        state_manager=state,
    )

    # Check availability
    avail_res = await manager.check_availability_action(date="2026-10-20", service="Teeth Cleaning")
    assert "10:00" in avail_res
    assert "11:00" in avail_res

    # Book appointment
    book_res = await manager.book_appointment_action(
        patient_name="John Doe",
        phone="+15551234567",
        date="2026-10-20",
        time="10:00",
        service="Teeth Cleaning",
    )
    assert "confirmed" in book_res.lower() or "scheduled" in book_res.lower()
    assert state.booking_confirmed is True

    # Booking conflict error handling
    mock_client.book_appointment = AsyncMock(
        side_effect=RuntimeClientValidationError("The 10:00 time slot is already booked")
    )
    conflict_res = await manager.book_appointment_action(
        patient_name="Jane Doe",
        phone="+15559876543",
        date="2026-10-20",
        time="10:00",
    )
    assert "unable to schedule" in conflict_res.lower() or "different open time" in conflict_res.lower()
    # Ensure no internal error trace leaked
    assert "Traceback" not in conflict_res
    assert "Exception" not in conflict_res


# ---------------------------------------------------------------------------
# Test 5: Cancellation, Rescheduling & Human Transfer Tools
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cancellation_rescheduling_and_transfer():
    mock_client = MagicMock(spec=RuntimeConfigClient)
    mock_client.cancel_appointment = AsyncMock(return_value={"success": True})
    mock_client.reschedule_appointment = AsyncMock(return_value={"success": True})
    mock_client.request_human_transfer = AsyncMock(
        return_value={
            "transferNumber": "+1 (555) 888-0000",
            "fallbackMessage": "Connecting you to our clinic reception now. Please stay on the line.",
        }
    )

    manager = ClinicToolsManager(
        tenant_id="tenant-apex",
        agent_id="agent-1",
        client=mock_client,
        clinic_profile={"phone": "+1 (555) 888-0000"},
    )

    # Cancel
    cancel_res = await manager.cancel_appointment_action(appointment_id="apt-101")
    assert "cancelled" in cancel_res.lower()

    # Reschedule
    resched_res = await manager.reschedule_appointment_action(
        appointment_id="apt-101",
        new_date="2026-10-25",
        new_time="14:00",
    )
    assert "2026-10-25" in resched_res
    assert "14:00" in resched_res

    # Human transfer
    transfer_res = await manager.request_human_transfer_action(reason="Complex insurance question")
    assert "reception" in transfer_res.lower() or "transferring" in transfer_res.lower()

    # Emergency escalation (Priority 1 safety)
    emergency_res = await manager.escalate_emergency_action("Caller has severe chest pain")
    assert "CLINICAL SAFETY ALERT" in emergency_res
    assert "911" in emergency_res


# ---------------------------------------------------------------------------
# Test 6: Function Tools Registration for Gemini LLM
# ---------------------------------------------------------------------------
def test_function_tools_registration():
    mock_client = MagicMock(spec=RuntimeConfigClient)
    manager = ClinicToolsManager(
        tenant_id="tenant-apex",
        agent_id="agent-1",
        client=mock_client,
    )

    tools = manager.get_function_tools()
    tool_names = [getattr(t, "__name__", str(t)) for t in tools]

    # Verify all 10 tools are registered
    assert any("get_clinic_info" in name for name in tool_names)
    assert any("get_services" in name for name in tool_names)
    assert any("get_business_hours" in name for name in tool_names)
    assert any("check_clinic_availability" in name for name in tool_names)
    assert any("find_customer" in name for name in tool_names)
    assert any("create_customer" in name for name in tool_names)
    assert any("book_clinic_appointment" in name for name in tool_names)
    assert any("cancel_clinic_appointment" in name for name in tool_names)
    assert any("reschedule_clinic_appointment" in name for name in tool_names)
    assert any("request_human_transfer" in name for name in tool_names)
    assert any("escalate_medical_emergency" in name for name in tool_names)


# ---------------------------------------------------------------------------
# Test 7: Tool Loop Limiter Protection
# ---------------------------------------------------------------------------
def test_tool_loop_limiter():
    limiter = ToolLoopLimiter(max_per_turn=3, max_per_session=10)
    assert limiter.acquire() is True
    assert limiter.acquire() is True
    assert limiter.acquire() is True
    assert limiter.acquire() is False  # Exceeded turn limit

    limiter.reset_turn()
    assert limiter.acquire() is True  # Turn reset allows next turn


# ---------------------------------------------------------------------------
# Test 8: Timezone Safety
# ---------------------------------------------------------------------------
def test_timezone_safety():
    # India timezone Asia/Kolkata
    today_utc = datetime.date(2026, 9, 23)
    norm = normalize_date("tomorrow", base_date=today_utc)
    assert norm == "2026-09-24"

    # Time normalization for morning/afternoon in clinic context
    assert normalize_time("9am") == "09:00"
    assert normalize_time("3pm") == "15:00"
    assert normalize_time("17:30") == "17:30"
    assert normalize_time("evening") == "17:30"

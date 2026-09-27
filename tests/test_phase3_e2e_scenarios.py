"""Phase 3 End-to-End Acceptance Tests.

Executes all 7 specified Phase 3 E2E test scenarios:
Scenario 1: Services Inquiry ("Hi, what services do you offer?") -> AI retrieves real configured services.
Scenario 2: Business Hours Inquiry ("What are your Saturday hours?") -> AI retrieves real business hours.
Scenario 3: Complete Booking Flow ("Can I book a cleaning appointment tomorrow?") -> Service + Customer + Availability + Confirmation + Transactional Booking.
Scenario 4: Cancellation Flow ("I want to cancel my appointment") -> Identify + Confirm + Backend Cancellation + Confirmation.
Scenario 5: Unavailable Time Slot Rejection -> No fabrication of availability.
Scenario 6: Cross-Tenant Security Isolation -> Tenant A strictly isolated from Tenant B.
Scenario 7: Duplicate Booking / Idempotency Retry -> No duplicate appointments created.
"""

import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from zcloud_ai_telecaller.clinic_tools import (
    ClinicToolsManager,
    normalize_date,
    normalize_time,
)
from zcloud_ai_telecaller.conversation_state import ConversationStateManager
from zcloud_ai_telecaller.prompt_builder import (
    build_clinic_system_prompt,
    match_deterministic_clinic_query,
)
from zcloud_ai_telecaller.runtime_client import (
    AgentRuntimeConfig,
    CallSessionAccumulator,
    RuntimeClientValidationError,
    RuntimeConfigClient,
)


@pytest.fixture
def mock_multitenant_backend():
    """Simulates the authoritative multi-tenant Control Plane database."""
    tenants = {
        "tenant_A": {
            "profile": {
                "clinicName": "Apex Dental Clinic",
                "phone": "+1 (555) 111-2222",
                "address": "123 Main St, Springfield",
                "timezone": "America/New_York",
            },
            "services": [
                {"id": "srv-1", "name": "Dental Cleaning & Checkup", "durationMinutes": 45},
                {"id": "srv-2", "name": "Routine Consultation", "durationMinutes": 30},
            ],
            "hours": {
                "monday": {"open": "08:00", "close": "18:00", "closed": False},
                "saturday": {"open": "09:00", "close": "13:00", "closed": False},
                "sunday": {"open": "00:00", "close": "00:00", "closed": True},
            },
            "customers": {
                "+15551234567": {"id": "cust-A1", "name": "Alice Springs", "phone": "+15551234567"},
            },
            "appointments": {},  # id -> record
            "booked_slots": set(),  # "date_time"
        },
        "tenant_B": {
            "profile": {
                "clinicName": "Shelbyville Medical Practice",
                "phone": "+1 (555) 333-4444",
                "address": "456 Oak St, Shelbyville",
                "timezone": "America/Chicago",
            },
            "services": [
                {"id": "srv-3", "name": "Cardiology Consultation", "durationMinutes": 60},
            ],
            "hours": {
                "saturday": {"open": "10:00", "close": "12:00", "closed": False},
                "sunday": {"open": "00:00", "close": "00:00", "closed": True},
            },
            "customers": {
                "+15559998888": {"id": "cust-B1", "name": "Bob Vance", "phone": "+15559998888"},
            },
            "appointments": {},
            "booked_slots": set(),
        },
    }

    client = MagicMock(spec=RuntimeConfigClient)

    async def get_clinic_info(tenant_id):
        return tenants[tenant_id]["profile"]

    async def get_services(tenant_id):
        return tenants[tenant_id]["services"]

    async def get_business_hours(tenant_id):
        return tenants[tenant_id]["hours"]

    async def check_availability(tenant_id, date, service_name=None):
        t_data = tenants[tenant_id]
        all_slots = ["09:00", "10:00", "11:00", "14:00", "15:00"]
        booked = t_data["booked_slots"]
        available = [s for s in all_slots if f"{date}_{s}" not in booked]
        return {
            "isOpen": True,
            "date": date,
            "availableSlots": available,
            "totalAvailable": len(available),
        }

    async def find_customer(tenant_id, phone=None, email=None):
        if phone in tenants[tenant_id]["customers"]:
            return tenants[tenant_id]["customers"][phone]
        return None

    async def create_customer(tenant_id, name, phone, email=None, notes=None):
        cust = {"id": f"cust_{len(tenants[tenant_id]['customers']) + 1}", "name": name, "phone": phone}
        tenants[tenant_id]["customers"][phone] = cust
        return {"customer": cust, "isExisting": False}

    async def book_appointment(tenant_id, **kwargs):
        date = kwargs.get("date")
        time_slot = kwargs.get("time")
        key = f"{date}_{time_slot}"
        if key in tenants[tenant_id]["booked_slots"]:
            raise RuntimeClientValidationError(f"Slot {time_slot} on {date} is already booked.")
        tenants[tenant_id]["booked_slots"].add(key)
        apt_id = f"apt_{tenant_id}_{len(tenants[tenant_id]['appointments']) + 1}"
        record = {
            "appointmentId": apt_id,
            "date": date,
            "time": time_slot,
            "status": "UPCOMING",
            "serviceName": kwargs.get("serviceName"),
            "patientName": kwargs.get("patientName"),
        }
        tenants[tenant_id]["appointments"][apt_id] = record
        return record

    async def cancel_appointment(tenant_id, appointment_id, reason=None):
        if appointment_id not in tenants[tenant_id]["appointments"]:
            raise RuntimeClientValidationError(f"Appointment {appointment_id} not found.")
        record = tenants[tenant_id]["appointments"][appointment_id]
        key = f"{record['date']}_{record['time']}"
        tenants[tenant_id]["booked_slots"].discard(key)
        record["status"] = "CANCELLED"
        return {"success": True, "appointmentId": appointment_id}

    async def reschedule_appointment(tenant_id, appointment_id, new_date, new_time):
        if appointment_id not in tenants[tenant_id]["appointments"]:
            raise RuntimeClientValidationError(f"Appointment {appointment_id} not found.")
        new_key = f"{new_date}_{new_time}"
        if new_key in tenants[tenant_id]["booked_slots"]:
            raise RuntimeClientValidationError(f"Slot {new_time} is already booked.")
        record = tenants[tenant_id]["appointments"][appointment_id]
        old_key = f"{record['date']}_{record['time']}"
        tenants[tenant_id]["booked_slots"].discard(old_key)
        tenants[tenant_id]["booked_slots"].add(new_key)
        record["date"] = new_date
        record["time"] = new_time
        record["status"] = "RESCHEDULED"
        return {"success": True, "appointmentId": appointment_id, "newDate": new_date, "newTime": new_time}

    client.get_clinic_info = AsyncMock(side_effect=get_clinic_info)
    client.get_services = AsyncMock(side_effect=get_services)
    client.get_business_hours = AsyncMock(side_effect=get_business_hours)
    client.check_availability = AsyncMock(side_effect=check_availability)
    client.find_customer = AsyncMock(side_effect=find_customer)
    client.create_customer = AsyncMock(side_effect=create_customer)
    client.book_appointment = AsyncMock(side_effect=book_appointment)
    client.cancel_appointment = AsyncMock(side_effect=cancel_appointment)
    client.reschedule_appointment = AsyncMock(side_effect=reschedule_appointment)

    return client, tenants


# ---------------------------------------------------------------------------
# E2E Scenario 1: Services Inquiry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_1_services_inquiry(mock_multitenant_backend):
    client, _ = mock_multitenant_backend
    manager = ClinicToolsManager(tenant_id="tenant_A", agent_id="agent-1", client=client)

    response = await manager.get_services_action()
    assert "Dental Cleaning & Checkup" in response
    assert "Routine Consultation" in response


# ---------------------------------------------------------------------------
# E2E Scenario 2: Business Hours Inquiry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_2_business_hours_inquiry(mock_multitenant_backend):
    client, _ = mock_multitenant_backend
    manager = ClinicToolsManager(tenant_id="tenant_A", agent_id="agent-1", client=client)

    response = await manager.get_business_hours_action()
    assert "Saturday: 09:00 to 13:00" in response
    assert "Sunday: Closed" in response


# ---------------------------------------------------------------------------
# E2E Scenario 3: Complete Booking Flow
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_3_complete_booking_flow(mock_multitenant_backend):
    client, tenants = mock_multitenant_backend
    state = ConversationStateManager(tenant_id="tenant_A", agent_id="agent-1")
    manager = ClinicToolsManager(tenant_id="tenant_A", agent_id="agent-1", client=client, state_manager=state)

    # 1. Check availability for tomorrow
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    avail = await manager.check_availability_action(date=tomorrow, service="Dental Cleaning & Checkup")
    assert "open appointments" in avail.lower()
    assert "10:00" in avail

    # 2. Register customer
    cust_res = await manager.create_customer_action(name="Alice Springs", phone="+1 (555) 123-4567")
    assert "Alice Springs" in cust_res

    # 3. Confirm & Book slot
    book_res = await manager.book_appointment_action(
        patient_name="Alice Springs",
        phone="+1 (555) 123-4567",
        date=tomorrow,
        time="10:00",
        service="Dental Cleaning & Checkup",
    )
    assert "scheduled" in book_res.lower() or "confirmed" in book_res.lower()
    assert state.booking_confirmed is True

    # Authoritative verification in store
    assert f"{tomorrow}_10:00" in tenants["tenant_A"]["booked_slots"]


# ---------------------------------------------------------------------------
# E2E Scenario 4: Cancellation Flow
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_4_cancellation_flow(mock_multitenant_backend):
    client, tenants = mock_multitenant_backend
    manager = ClinicToolsManager(tenant_id="tenant_A", agent_id="agent-1", client=client)

    # Pre-book appointment
    test_date = "2026-10-15"
    book_res = await client.book_appointment(
        "tenant_A",
        date=test_date,
        time="11:00",
        patientName="Alice Springs",
        serviceName="Cleaning",
    )
    apt_id = book_res["appointmentId"]
    assert f"{test_date}_11:00" in tenants["tenant_A"]["booked_slots"]

    # Cancel via tool
    cancel_res = await manager.cancel_appointment_action(appointment_id=apt_id, reason="Schedule conflict")
    assert "cancelled" in cancel_res.lower()
    assert "released" in cancel_res.lower()

    # Verify slot is freed in store
    assert f"{test_date}_11:00" not in tenants["tenant_A"]["booked_slots"]
    assert tenants["tenant_A"]["appointments"][apt_id]["status"] == "CANCELLED"


# ---------------------------------------------------------------------------
# E2E Scenario 5: Unavailable Time Slot Rejection
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_5_unavailable_slot_rejection(mock_multitenant_backend):
    client, _ = mock_multitenant_backend
    manager = ClinicToolsManager(tenant_id="tenant_A", agent_id="agent-1", client=client)

    test_date = "2026-10-16"
    # Pre-book 14:00
    await client.book_appointment(
        "tenant_A",
        date=test_date,
        time="14:00",
        patientName="Patient One",
        serviceName="Checkup",
    )

    # Attempt to book the same slot
    conflict_res = await manager.book_appointment_action(
        patient_name="Patient Two",
        phone="+15559990000",
        date=test_date,
        time="14:00",
    )
    assert "unable to schedule" in conflict_res.lower() or "different open time" in conflict_res.lower()


# ---------------------------------------------------------------------------
# E2E Scenario 6: Multi-Tenant Security Isolation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_6_cross_tenant_isolation(mock_multitenant_backend):
    client, _ = mock_multitenant_backend
    manager_A = ClinicToolsManager(tenant_id="tenant_A", agent_id="agent-1", client=client)
    manager_B = ClinicToolsManager(tenant_id="tenant_B", agent_id="agent-2", client=client)

    # Tenant A sees Dental Cleaning
    serv_A = await manager_A.get_services_action()
    assert "Dental Cleaning" in serv_A
    assert "Cardiology" not in serv_A

    # Tenant B sees Cardiology
    serv_B = await manager_B.get_services_action()
    assert "Cardiology" in serv_B
    assert "Dental Cleaning" not in serv_B

    # Customer lookup isolation
    find_B_in_A = await manager_A.find_customer_action(phone="+15559998888")  # Bob Vance is in Tenant B
    assert "could not find" in find_B_in_A.lower()


# ---------------------------------------------------------------------------
# E2E Scenario 7: Idempotency & Duplicate Prevention
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_7_duplicate_booking_prevention(mock_multitenant_backend):
    client, tenants = mock_multitenant_backend
    manager = ClinicToolsManager(tenant_id="tenant_A", agent_id="agent-1", client=client)

    test_date = "2026-10-22"
    res1 = await manager.book_appointment_action(
        patient_name="Clark Kent",
        phone="+15553337777",
        date=test_date,
        time="09:00",
        service="General Consultation",
    )
    assert "scheduled" in res1.lower() or "confirmed" in res1.lower()

    # Same slot retry is blocked
    res2 = await manager.book_appointment_action(
        patient_name="Clark Kent",
        phone="+15553337777",
        date=test_date,
        time="09:00",
        service="General Consultation",
    )
    assert "unable to schedule" in res2.lower()
    # Ensure exactly 1 booked slot in backend
    assert len([s for s in tenants["tenant_A"]["booked_slots"] if s == f"{test_date}_09:00"]) == 1

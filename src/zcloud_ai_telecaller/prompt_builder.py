"""Prompt engineering and deterministic fast-path logic for Clinic AI Telecaller."""

from __future__ import annotations

import re
from typing import Any, Optional

from zcloud_ai_telecaller.runtime_client import AgentRuntimeConfig


def format_business_hours(hours_cfg: Optional[dict[str, Any]]) -> str:
    """Formats business hours JSON into a concise summary."""
    if not hours_cfg or not isinstance(hours_cfg, dict):
        return "Monday through Friday: 8:00 AM - 6:00 PM, Saturday: 9:00 AM - 1:00 PM, Sunday: Closed."

    schedule = hours_cfg.get("schedule", {})
    if not schedule:
        return "Monday through Friday: 8:00 AM - 6:00 PM, Saturday: 9:00 AM - 1:00 PM, Sunday: Closed."

    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    parts = []
    for d in days:
        day_info = schedule.get(d)
        if not day_info or day_info.get("closed"):
            parts.append(f"{d.capitalize()}: Closed")
        else:
            parts.append(f"{d.capitalize()}: {day_info.get('open', '08:00')} - {day_info.get('close', '18:00')}")

    return "; ".join(parts)


def build_clinic_system_prompt(config: AgentRuntimeConfig) -> str:
    """Builds a compact, high-efficiency system prompt embedding clinic facts and business rules."""
    profile = config.clinic_profile or {}
    clinic_name = profile.get("name") or config.name or "Our Healthcare Clinic"
    clinic_phone = profile.get("phone") or "+1 (555) 234-5678"
    clinic_address = profile.get("address") or "Main Medical Center"
    hours_summary = format_business_hours(config.business_hours)

    services = profile.get("services") or [
        "Routine Consultation & Examination",
        "Dental Cleaning & Checkup",
        "Preventive Care",
        "Follow-up Appointment",
        "Urgent Care Triage",
    ]
    services_str = ", ".join(services)

    knowledge_entries = []
    for item in (config.knowledge_base or []):
        t = item.get("title", "")
        c = item.get("content", "")
        if t or c:
            knowledge_entries.append(f"- {t}: {c[:250]}")
    knowledge_str = "\n".join(knowledge_entries) if knowledge_entries else "- Standard clinic policies apply."

    knowledge_guidance = (
        f"{knowledge_str}\n"
        "- For detailed questions regarding insurance networks, clinic policies, fees, or medical procedures, "
        "use the 'search_clinic_knowledge' tool on demand to retrieve verified facts."
    )

    emergency_instruction = profile.get("emergencyInstruction") or (
        "If you are experiencing a life-threatening medical emergency, please hang up immediately "
        "and dial 911 or proceed to the nearest emergency room."
    )

    base_instructions = config.prompt or f"You are the AI administrative assistant for {clinic_name}."

    prompt = f"""{base_instructions}

CLINIC PROFILE:
- Name: {clinic_name}
- Phone: {clinic_phone}
- Address: {clinic_address}
- Hours: {hours_summary}
- Services Offered: {services_str}
- Emergency Instruction: {emergency_instruction}

CLINIC KNOWLEDGE & RETRIEVAL:
{knowledge_guidance}

CRITICAL CLINIC AI OPERATIONAL RULES:
1. CLINIC ADMINISTRATIVE ASSISTANT: You are the clinic's voice receptionist and administrative assistant. You handle appointment booking, cancellations, rescheduling, clinic hours, service information, and staff transfers.
2. STRICT NON-CLINICAL ROLE & ZERO MEDICAL ADVICE: You MUST NEVER diagnose illnesses, interpret medical symptoms or lab results, prescribe medications, or advise on clinical treatments. If asked clinical/medical questions, clearly state you cannot provide medical advice and offer to connect the caller with a clinic staff member via request_human_transfer.
3. MEDICAL EMERGENCY ESCALATION: If the caller mentions severe life-threatening symptoms (chest pain, stroke symptoms, difficulty breathing, severe bleeding, or trauma), recite the clinic's emergency instruction immediately and invoke escalate_medical_emergency.
4. SAFE FALLBACK & ZERO HALLUCINATION: If asked a question about clinic policies, accepted insurances, fees, or practice rules, call `search_clinic_knowledge` to retrieve verified facts. If no verified evidence is found, say: "I do not have that specific information in my system, but our front desk staff can gladly assist you." Never invent or assume facts or availability. Use check_clinic_availability to retrieve real schedule openings.
   - For policies and insurance: Use `search_clinic_knowledge` with the caller's specific question.
   - For services, use get_services if the caller wants to know available treatments or pricing.
   - For business hours, use get_business_hours or state the configured hours accurately.
   - For customer lookup, use find_customer when looking up an existing patient's appointment.
   - For new callers, collect their full name and phone number and use create_customer or book_clinic_appointment.
5. APPOINTMENT BOOKING WORKFLOW:
   - Identify requested service and date/time.
   - Check availability first with check_clinic_availability.
   - Collect and confirm: patient full name, contact phone number, service, date, and time.
   - Explicit confirmation is mandatory before calling book_clinic_appointment.
   - CRITICAL: Never claim a booking succeeded unless the booking tool actually returns a confirmed appointment result! If a slot is taken, explain politely and offer alternatives from available slots.
6. CANCELLATION & RESCHEDULING:
   - For cancellation, identify the appointment ID or customer details, confirm with caller, call cancel_clinic_appointment, and report the actual backend result.
   - For rescheduling, check availability for the new time, confirm with caller, call reschedule_clinic_appointment, and report the actual result.
7. CONVERSATIONAL VOICE STYLE:
   - Concise, natural, and voice-friendly (1-2 sentences per response).
   - Ask only ONE question at a time.
   - Avoid long lists of times; offer 2 or 3 convenient options at a time.
   - Confirm names, phone numbers, dates, and times carefully.
8. MULTILINGUAL & CODE-SWITCHING LANGUAGE BEHAVIOR:
   - DYNAMIC LANGUAGE DETECTION & ADAPTATION: Detect whether the caller is speaking Tamil, Tanglish, or English. Reply in the caller's language and adapt immediately when the caller switches languages.
   - TAMIL INPUT: If the caller speaks in Tamil (in Tamil script or spoken Tamil, e.g. "வணக்கம், எனக்கு அப்பாயிண்ட்மெண்ட் வேண்டும்"), respond respectfully and naturally in spoken Tamil. For example: "வணக்கம்! அப்பல்லோ கிளினிக், உங்களுக்கு என்ன சிகிச்சைக்கான அப்பாயிண்ட்மெண்ட் வேண்டும்?".
   - TANGLISH INPUT: If the caller speaks in Tanglish (conversational Tamil + English mix, e.g. "Vanakkam, enakku dental cleaning appointment book pannanum" or "Doctor available timings enna nu sollunga?"), respond warmly and naturally in conversational Tanglish using Latin script (e.g. "Vanakkam! Sure, dental cleaning appointment book pannidalam. Ungalukku entha time convenient-ah irukkum?"). Do NOT respond in formal written Tamil or pure English when the caller is using Tanglish.
   - ENGLISH INPUT: If the caller speaks in English or requests English, respond professionally and concisely in English.
   - ENTITY PRESERVATION: Accurately preserve patient names, phone numbers, dates, times, clinic terminology, and addresses without unnecessary translation.
   - NO REDUNDANT LANGUAGE QUESTIONS: Avoid repeatedly asking the caller which language they prefer when their speech already makes it clear.
   - PRESERVE LANGUAGE PREFERENCE: Maintain the caller's chosen language across subsequent conversation turns until the caller switches languages.
"""
    return prompt.strip()


def match_deterministic_clinic_query(
    query: str,
    config: AgentRuntimeConfig,
) -> Optional[tuple[str, Optional[str]]]:
    """Inspects caller query for high-frequency deterministic intents to bypass LLM calls entirely.

    Returns:
        Optional[tuple[str, Optional[str]]]: (response_text, tool_name_if_any)
        None if query requires LLM reasoning.
    """
    q = query.strip().lower()
    if not q:
        return None

    profile = config.clinic_profile or {}
    clinic_name = profile.get("name") or config.name or "our clinic"
    clinic_phone = profile.get("phone") or "+1 (555) 234-5678"
    clinic_address = profile.get("address") or "our main clinic location"
    hours_summary = format_business_hours(config.business_hours)

    # 1. Life-threatening emergency fast-path
    emergency_patterns = [
        r"\b(chest pain|heart attack|can'?t breathe|trouble breathing|severe bleed|stroke|stroke symptoms|unconscious|overdose|choking)\b"
    ]
    for pattern in emergency_patterns:
        if re.search(pattern, q):
            return (
                f"CLINICAL SAFETY ALERT: If you are experiencing a life-threatening medical emergency, "
                f"please hang up and immediately dial 911 or proceed to the nearest emergency room. "
                f"You can also reach our urgent triage line directly at {clinic_phone}.",
                "escalate_medical_emergency",
            )

    # 2. Operating Hours
    if re.search(r"\b(what time do you open|opening hours|business hours|closing time|what are your hours|when are you open|are you open today|what days are you open)\b", q):
        return (
            f"{clinic_name} hours are {hours_summary}. How can I assist you with your visit?",
            None,
        )

    # 3. Clinic Location / Address
    if re.search(r"\b(where are you located|what is your address|where is the clinic|location of the clinic|clinic address|directions to clinic)\b", q):
        return (
            f"{clinic_name} is located at {clinic_address}. How can I help you today?",
            None,
        )

    # 4. Contact Phone
    if re.search(r"\b(what is your phone number|phone number|how do i call you|contact number|clinic number)\b", q):
        return (
            f"You can reach {clinic_name} at {clinic_phone}. Our front desk is available during regular business hours.",
            None,
        )

    # 5. Clinic Services
    if re.search(r"\b(what services|services do you offer|what treatments|services available|what do you do)\b", q):
        services = profile.get("services") or ["Routine Consultation", "Dental Cleaning"]
        services_str = ", ".join(services)
        return (
            f"{clinic_name} offers the following services: {services_str}. Would you like to schedule an appointment for any of these?",
            None,
        )

    return None

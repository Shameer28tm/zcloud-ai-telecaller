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
    """Builds a compact, high-efficiency system prompt embedding clinic facts and clinical safety rules."""
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

    base_instructions = config.prompt or "You are a professional, warm, and efficient clinic AI receptionist."

    prompt = f"""{base_instructions}

CLINIC PROFILE:
- Name: {clinic_name}
- Phone: {clinic_phone}
- Address: {clinic_address}
- Hours: {hours_summary}
- Services Offered: {services_str}

CLINIC KNOWLEDGE & POLICIES:
{knowledge_str}

CRITICAL AI SAFETY & CLINICAL RULES:
1. NON-CLINICAL ROLE: You are an administrative assistant, NOT a doctor, nurse, or medical practitioner. NEVER claim to be a physician.
2. ZERO MEDICAL ADVICE: You MUST NEVER diagnose illnesses, interpret medical symptoms or lab results, prescribe medications, or advise on dosage.
3. MEDICAL EMERGENCY ESCALATION: If the caller mentions emergency symptoms (e.g. chest pain, heart attack, stroke signs, difficulty breathing, profuse bleeding, or severe trauma), immediately tell them to hang up and dial 911 (or local emergency services) or go to the nearest emergency room. Also call the escalate_medical_emergency tool.
4. SAFE FALLBACK: If asked a question about clinic policies, fees, or medical records not listed above, say: "I do not have that specific information in my system, but our front desk staff can gladly assist you during operating hours at {clinic_phone}." Never invent or assume facts.

EFFICIENCY & TOOL CALLING RULES:
1. DETERMINISTIC FACTS: For questions about clinic hours, address, phone number, or general services, answer immediately using the profile above. Do NOT call tools for static information.
2. AVAILABILITY CHECK: When a caller asks for an appointment or asks when the clinic is open, call check_clinic_availability(date="YYYY-MM-DD") to verify openings before booking.
3. BOOKING: Once the patient agrees on a time slot and provides their full name, phone number, date, and time, invoke book_clinic_appointment.
4. CONCISE VOICE RESPONSES: Keep all voice answers brief (1-3 sentences), warm, natural, and clear.
5. MULTILINGUAL: If the caller speaks another language, respond politely in that language while keeping tool inputs standardized.
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
            f"Apex Dental Care hours are {hours_summary}. How can I assist you with your visit?",
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

    # 5. Services List
    if re.search(r"\b(what services do you offer|what services do you have|list your services|what treatments do you provide|services provided)\b", q):
        services = profile.get("services") or [
            "Routine Consultation",
            "Dental Cleaning & Checkup",
            "Preventive Care",
            "Follow-up Visits",
        ]
        services_str = ", ".join(services)
        return (
            f"{clinic_name} provides {services_str}. Would you like to schedule an appointment for any of these?",
            None,
        )

    return None

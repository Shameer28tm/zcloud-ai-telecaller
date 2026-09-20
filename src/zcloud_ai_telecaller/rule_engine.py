"""Tenant-Aware Business Rule Engine for Z Cloud AI Telecaller.

Enforces strict priority hierarchy:
1. Safety Rules (Non-clinical role, emergency escalation, no diagnosis/prescribing)
2. Security Rules (Multi-tenant isolation, prompt injection defense, secret protection)
3. Tenant Business Rules (Required confirmation, tool-required intents, operating hours)
4. Tool Correctness (Strict schema parameters, validated date/time)
5. Conversation Policy (Bounded turns, barge-in enabled, context limits)
6. Knowledge & Context (Clinic facts, hours, services)
7. LLM Style & Tone (Concise receptionist, warm tone)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger("zcloud_ai_telecaller.rule_engine")


class RuleEvaluationResult:
    """Result of pre-turn or tool rule evaluation."""

    def __init__(
        self,
        action: str,  # "PASS", "FAST_PATH", "ESCALATE", "BLOCK", "PROMPT_CONFIRMATION"
        message: Optional[str] = None,
        tool_name: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        self.action = action
        self.message = message
        self.tool_name = tool_name
        self.reason = reason

    @property
    def is_blocking(self) -> bool:
        return self.action in ("ESCALATE", "BLOCK", "PROMPT_CONFIRMATION", "FAST_PATH")


class BusinessRuleEngine:
    """Evaluates user requests and actions against the 7-level rule hierarchy."""

    def __init__(self, rules_config: Optional[dict[str, Any]] = None) -> None:
        self.rules = rules_config or {}
        self.safety_rules = self.rules.get("safetyRules", {})
        self.security_rules = self.rules.get("securityRules", {})
        self.business_rules = self.rules.get("businessRules", {})
        self.conversation_policy = self.rules.get("conversationPolicy", {})
        self.style_and_tone = self.rules.get("styleAndTone", {})

        # Life-threatening emergency keywords (Immutable Safety Layer 1)
        self.emergency_patterns = [
            r"\b(chest pain|heart attack|can'?t breathe|trouble breathing|severe bleed|profuse bleed|stroke|unconscious|head trauma|overdose|choking)\b"
        ]

        # Prompt injection & jailbreak patterns (Security Layer 2)
        self.injection_patterns = [
            r"\b(ignore\s+(all\s+)?(previous\s+)?instructions|ignore\s+your\s+rules)\b",
            r"\b(system\s+prompt|reveal\s+prompt|show\s+prompt|hidden\s+prompt)\b",
            r"\b(api\s+key|worker\s+secret|secret\s+key|admin\s+password)\b",
            r"\b(bypass\s+confirmation|book\s+me\s+without\s+confirmation)\b",
            r"\b(show\s+me\s+another\s+(patient|customer)|other\s+patient('s)?\s+data)\b",
        ]

    def evaluate_pre_turn(
        self,
        user_query: str,
        clinic_phone: str = "+1 (555) 234-5678",
    ) -> RuleEvaluationResult:
        """Evaluates an incoming caller query against safety, security, and policy rules BEFORE any LLM invocation."""
        q = user_query.strip().lower()
        if not q:
            return RuleEvaluationResult("PASS")

        # -------------------------------------------------------------
        # PRIORITY 1: Safety Rules (Emergency Escalation)
        # -------------------------------------------------------------
        for pattern in self.emergency_patterns:
            if re.search(pattern, q):
                handoff = self.safety_rules.get(
                    "emergencyHandoffMessage",
                    "CLINICAL SAFETY ALERT: If you are experiencing a life-threatening medical emergency, "
                    "please hang up immediately and dial 911 (or local emergency services) or proceed to the nearest emergency room."
                )
                msg = f"{handoff} You can also reach our urgent triage directly at {clinic_phone}."
                logger.warning("Priority 1 Safety Rule triggered for query: '%s'", user_query)
                return RuleEvaluationResult(
                    action="ESCALATE",
                    message=msg,
                    tool_name="escalate_medical_emergency",
                    reason="Life-threatening emergency symptom detected",
                )

        # -------------------------------------------------------------
        # PRIORITY 2: Security Rules (Prompt Injection & Secret Protection)
        # -------------------------------------------------------------
        for pattern in self.injection_patterns:
            if re.search(pattern, q):
                logger.warning("Priority 2 Security Rule triggered (Prompt Injection): '%s'", user_query)
                refusal = (
                    "I am an administrative clinic assistant. I cannot modify my system instructions, "
                    "disclose system credentials, or access other patients' private records. "
                    "How may I assist you with scheduling or clinic inquiries?"
                )
                return RuleEvaluationResult(
                    action="BLOCK",
                    message=refusal,
                    reason="Prompt injection / unauthorized security attempt blocked",
                )

        # -------------------------------------------------------------
        # PRIORITY 3: Tenant Business Rules (Medical Advice Prohibition)
        # -------------------------------------------------------------
        medical_advice_pattern = r"\b(can you diagnose|prescribe me|what medication should i take|am i sick with|dosage for)\b"
        if re.search(medical_advice_pattern, q):
            advice_refusal = (
                "As an administrative voice assistant, I cannot diagnose medical conditions or prescribe medications. "
                "I would be glad to schedule an in-person consultation with our healthcare team so a doctor can evaluate you."
            )
            return RuleEvaluationResult(
                action="BLOCK",
                message=advice_refusal,
                reason="Medical advice request blocked",
            )

        return RuleEvaluationResult("PASS")

    def validate_booking_authorization(
        self,
        user_query: str,
        is_availability_question_only: bool,
    ) -> bool:
        """Enforces that asking about availability != booking authorization.

        Requires affirmative confirmation before invoking booking.
        """
        q = user_query.strip().lower()

        # Merely asking if a slot is available
        if is_availability_question_only or re.search(r"\b(is|are)\s+.*(available|open|free)\b", q):
            return False

        # Positive confirmation expressions
        affirmative_patterns = [
            r"\b(yes|yeah|sure|please\s+book|book\s+it|confirm|that\s+works|sounds\s+good|go\s+ahead|okay|ok)\b",
            r"\b(book\s+for|schedule\s+for|i'll\s+take)\b",
        ]
        return any(re.search(pat, q) for pat in affirmative_patterns)

    def sanitize_response(self, text: str) -> str:
        """Enforces concise response length and strips any accidental credential leaks."""
        # Strip potential secrets
        cleaned = re.sub(r"(zca_worker_[a-zA-Z0-9_]+|sk-[a-zA-Z0-9]+|bearer\s+[a-zA-Z0-9_]+)", "[REDACTED]", text, flags=re.IGNORECASE)

        # Enforce max word count if specified (default: 45 words for conversational phone telecalling)
        max_words = self.business_rules.get("maxResponseLengthWords", 45)
        words = cleaned.split()
        if len(words) > max_words + 10:
            truncated = " ".join(words[:max_words]) + "..."
            return truncated

        return cleaned

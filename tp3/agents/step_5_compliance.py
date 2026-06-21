"""5.1 Compliance PII Agent - anonymization & reidentification-risk officer.

Acts as a privacy/compliance officer for the run. It de-identifies the case
material so it can be safely reused for learning, aligned with the HIPAA
Safe-Harbor 18 identifiers and Quebec Law 25 "renseignements personnels". It
never alters clinical facts - it only redacts identifiers - and it estimates the
residual reidentification risk from remaining quasi-identifiers. It refuses to
approve when anonymization cannot be guaranteed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents.base_agent import AgentResponse, BaseAgent
from config import settings
from core import llm
from tools import mongodb_tools

AGENT_ID = "5.1 Compliance PII Agent"
LIKELY_NEXT = [
    "5.3 Reflection Agent",
    "1.1 Clinical Agent Orchestrator",
]

# HIPAA Safe-Harbor + Law 25 categories the agent is expected to screen for.
_IDENTIFIER_CATEGORIES = (
    "names; geographic subdivisions smaller than a state; all date elements "
    "(birth, admission, discharge, death) except year; phone/fax numbers; email "
    "addresses; SSN/SIN/NAS; medical record, health-plan, account, certificate, "
    "or license numbers; vehicle/device identifiers; URLs and IP addresses; "
    "biometric identifiers; full-face photos; and any other unique identifying "
    "number, characteristic, or code (HIPAA Safe Harbor). Under Quebec Law 25, "
    "also treat any information allowing a person to be identified directly or "
    "indirectly as 'renseignements personnels'."
)


class CompliancePIIOutput(BaseModel):
    identified_phi: list[str] = Field(
        default_factory=list,
        description="PHI / personal information found in the case material "
                    "(category + where it appears), without echoing the raw value.",
    )
    redactions: list[str] = Field(
        default_factory=list,
        description="What was masked and the placeholder token used "
                    "(e.g. 'patient name -> [NAME]').",
    )
    sanitized_summary: str = Field(
        default="",
        description="De-identified narrative of the case that preserves all "
                    "clinical facts but contains no direct or indirect identifiers.",
    )
    reidentification_risks: list[str] = Field(
        default_factory=list,
        description="Residual quasi-identifiers (rare diagnoses, exact ages, "
                    "small-population locations, dates) that could enable reidentification.",
    )
    reidentification_risk_score: float = Field(
        default=1.0,
        description="0 (no realistic risk) to 1 (easily reidentifiable).",
    )
    compliance_notes: list[str] = Field(default_factory=list)
    anonymization_guaranteed: bool = Field(
        default=False,
        description="True only if the sanitized_summary is safe to reuse for learning.",
    )
    next_agent: str = Field(default="5.3 Reflection Agent")
    handoff_reason: str = ""
    needs_orchestrator: bool = False


_SYSTEM = (
    "You are the 5.1 Compliance PII Agent, the privacy and compliance officer of "
    "an autonomous clinical multi-agent system. Your job is to make the case "
    "material safe to reuse for learning WITHOUT changing any clinical fact. "
    "Screen for these identifiers: "
    f"{_IDENTIFIER_CATEGORIES} "
    "Produce a de-identified narrative (sanitized_summary) that keeps the "
    "clinical reasoning intact but removes/masks every direct identifier, then "
    "assess the residual reidentification risk from remaining quasi-identifiers. "
    "Guardrails: NEVER invent or alter clinical facts; ONLY redact. NEVER echo a "
    "raw identifier value back - describe it by category. If you cannot guarantee "
    "anonymization, set anonymization_guaranteed=false and keep the risk score "
    "high. Be conservative: when in doubt, redact. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}."
)


def _case_material(state: dict) -> str:
    """Collect the run's textual content the officer must screen and sanitize."""
    fields = [
        ("Patient question", state.get("patient_question", "")),
        ("Intent", state.get("intent", "")),
        ("Final patient explanation", state.get("patient_explanation", "")),
        ("Key points", state.get("patient_key_points", [])),
        ("Recommended actions", state.get("patient_recommended_actions", [])),
        ("Timeline", state.get("timeline", [])),
        ("Risks", state.get("risks", [])),
        ("Red flags", state.get("red_flags", [])),
        ("Similar cases", state.get("similar_cases", [])),
        ("Care plan", state.get("care_plan", [])),
    ]
    parts: list[str] = []
    for label, value in fields:
        if not value:
            continue
        if isinstance(value, (list, dict)):
            try:
                rendered = json.dumps(value, ensure_ascii=False, default=str)[:1500]
            except Exception:
                rendered = str(value)[:1500]
        else:
            rendered = str(value)[:1500]
        parts.append(f"{label}: {rendered}")
    return "\n".join(parts) if parts else "(no case material gathered yet)"


class CompliancePIIAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "strong"

    def execute(self, state):
        user = (
            f"{self.context_block(state)}\n\n"
            "CASE MATERIAL TO SCREEN AND DE-IDENTIFY:\n"
            f"{_case_material(state)}\n\n"
            "Identify all PHI / personal information, produce the sanitized "
            "narrative, and assess reidentification risk now."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=[],
            schema=CompliancePIIOutput,
            tier=self.TIER,
            temperature=0.0,
        )

        risk_score = max(0.0, min(1.0, parsed.reidentification_risk_score))
        approved = bool(
            parsed.anonymization_guaranteed
            and parsed.sanitized_summary.strip()
            and risk_score < settings.REIDENTIFICATION_RISK_THRESHOLD
        )

        sanitized_case = {
            "summary": parsed.sanitized_summary,
            "redactions": parsed.redactions,
            "identified_phi": parsed.identified_phi,
        }

        memory_updates = {
            "sanitized_case": sanitized_case,
            "reidentification_risk": parsed.reidentification_risks,
            "reidentification_risk_score": risk_score,
            "compliance_notes": parsed.compliance_notes,
            "compliance_approved": approved,
            "compliance_reviewed": True,
        }

        # Persist a compliance audit record (MongoDB, not long-term memory).
        try:
            mongodb_tools.insert_audit_event(
                {
                    "event": "compliance_review",
                    "runId": state.get("run_id"),
                    "patientId": state.get("patient_id"),
                    "approved": approved,
                    "reidentificationRiskScore": risk_score,
                    "reidentificationRisks": parsed.reidentification_risks,
                    "identifiedPhiCount": len(parsed.identified_phi),
                    "redactions": parsed.redactions,
                    "notes": parsed.compliance_notes,
                }
            )
        except Exception:
            pass

        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed",
            memory_updates=memory_updates,
            next_agent=None if parsed.needs_orchestrator else parsed.next_agent,
            handoff_reason=parsed.handoff_reason
            or (
                "Case de-identified and approved for learning; reflection next."
                if approved
                else "Reidentification risk too high or anonymization not "
                "guaranteed; flagged. Reflection next."
            ),
            needs_orchestrator=parsed.needs_orchestrator,
        )
        return response, tool_records, token_records


run = CompliancePIIAgent()

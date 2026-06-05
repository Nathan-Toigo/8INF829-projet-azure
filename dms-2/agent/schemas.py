"""Structured output schemas for the 5 clinical steps.

Structured (Pydantic) step outputs keep the audit panel reliable and let later
steps consume earlier findings without re-parsing prose.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ContextResult(BaseModel):
    """Step 1 - Comprendre le contexte."""

    chronic_conditions: list[str] = Field(
        default_factory=list, description="Probable chronic diseases identified."
    )
    medications: list[str] = Field(
        default_factory=list, description="Current medications found in the chart."
    )
    longitudinal_portrait: str = Field(
        "", description="Narrative longitudinal portrait of the patient."
    )
    key_documents: list[str] = Field(
        default_factory=list, description="Source documents the context relies on."
    )


class RiskItem(BaseModel):
    description: str
    severity: str = Field("medium", description="low | medium | high")
    evidence: str = Field("", description="Chart evidence supporting the risk.")
    source: str = Field("", description="Source document for the evidence.")


class RiskResult(BaseModel):
    """Step 2 - Detecter les risques."""

    risks: list[RiskItem] = Field(default_factory=list)


class PriorityResult(BaseModel):
    """Step 3 - Prioriser."""

    urgent: list[str] = Field(default_factory=list)
    can_wait: list[str] = Field(default_factory=list)
    needs_clarification: list[str] = Field(default_factory=list)


class ActionResult(BaseModel):
    """Step 4 - Produire des actions."""

    physician_questions: list[str] = Field(default_factory=list)
    behavior_changes: list[str] = Field(default_factory=list)
    exams_to_request: list[str] = Field(default_factory=list)
    followup_reminders: list[str] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    """Step 5 - Generer une synthese patient."""

    patient_summary: str = Field("", description="Simplified, vulgarized summary.")
    language: str = Field("fr", description="Language code of the summary.")

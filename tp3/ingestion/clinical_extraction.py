"""Structured clinical-entity extraction from raw document text.

Turns free OCR/extracted text into the OCR output schema (spec section 15) and
FHIR-like ``clinical_resources`` documents (spec section 7).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from core import llm

DOCUMENT_TYPES = [
    "lab_report",
    "imaging_report",
    "consultation_note",
    "pathology_report",
    "discharge_summary",
    "medication_list",
    "other",
]


class ClinicalEntity(BaseModel):
    type: str = Field(
        default="Observation",
        description="FHIR-like resource type: Observation, Condition, "
        "MedicationStatement, AllergyIntolerance, Procedure, etc.",
    )
    name: str = Field(default="", description="Human-readable name, e.g. 'HbA1c'.")
    value: str = Field(default="", description="Measured value or status.")
    unit: str = Field(default="", description="Unit when applicable, e.g. '%'.")
    date: str = Field(default="", description="ISO date of the event if known.")
    code_system: str = Field(default="", description="e.g. LOINC, SNOMED.")
    code: str = Field(default="", description="Code value if identifiable.")


class ExtractionResult(BaseModel):
    document_type: str = Field(default="other")
    clinical_entities: list[ClinicalEntity] = Field(default_factory=list)
    confidence: float = Field(default=0.0)


_SYSTEM = (
    "You are a clinical information extraction engine. Given the raw text of a "
    "single medical document, classify the document type and extract structured "
    "clinical entities (observations/labs, conditions, medications, allergies, "
    "procedures) with values, units, and dates when present. Use ISO dates "
    "(YYYY-MM-DD) when a date is identifiable. Only extract what is explicitly "
    "stated. Provide an overall extraction confidence between 0 and 1. "
    f"document_type must be one of: {', '.join(DOCUMENT_TYPES)}."
)


def extract_entities(raw_text: str) -> tuple[ExtractionResult, dict]:
    """Run structured extraction over raw document text."""
    if not raw_text.strip():
        return ExtractionResult(), {"step": "clinical_extraction", "total_tokens": 0}
    user = f"Document text:\n\n{raw_text[:12000]}"
    parsed, usage = llm.invoke_structured(
        step="clinical_extraction",
        system=_SYSTEM,
        user=user,
        schema=ExtractionResult,
        tier="small",
    )
    return parsed, usage


def entities_to_resources(
    entities: list[ClinicalEntity], patient_id: str, source_document_id: str
) -> list[dict]:
    """Map extracted entities to FHIR-like ``clinical_resources`` docs."""
    resources: list[dict] = []
    for e in entities:
        resources.append(
            {
                "patientId": patient_id,
                "resourceType": e.type or "Observation",
                "code": {
                    "system": e.code_system or "",
                    "code": e.code or "",
                    "display": e.name or "",
                },
                "value": e.value or "",
                "unit": e.unit or "",
                "effectiveDate": e.date or "",
                "sourceDocumentId": source_document_id,
                "fhir": {},
            }
        )
    return resources

"""Per-run tool factories bound to a specific patient.

Each agent gets only the tools it needs (spec section 11). Tools close over the
current ``patient_id`` so the model never has to guess identifiers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.tools import StructuredTool

from config import settings
from tools import chromadb_tools, mongodb_tools, web_search_tools


def _format_hits(hits) -> str:
    if not hits:
        return "No matches found."
    lines = []
    for h in hits:
        meta = h.metadata or {}
        tag = meta.get("fileName") or meta.get("topic") or meta.get("source") or ""
        lines.append(f"[{h.rank}] ({h.similarity:.2f}) {tag}: {h.content[:500]}")
    return "\n\n".join(lines)


def make_patient_records_tool(patient_id: str) -> StructuredTool:
    def _fn(resource_type: str = "") -> str:
        """Fetch structured clinical resources (labs, conditions, medications,
        allergies, procedures) stored for this patient from MongoDB. Optionally
        filter by FHIR resourceType (e.g. 'Observation', 'Condition')."""
        resources = mongodb_tools.list_clinical_resources(
            patient_id, resource_type or None
        )
        if not resources:
            return "No structured clinical resources stored for this patient."
        lines = []
        for r in resources[:60]:
            code = (r.get("code") or {}).get("display", "")
            lines.append(
                f"- {r.get('resourceType')}: {code} = {r.get('value','')} "
                f"{r.get('unit','')} ({r.get('effectiveDate','') or 'no date'})"
            )
        return "\n".join(lines)

    return StructuredTool.from_function(
        _fn,
        name="mongodb_patient_records",
        description="Read structured clinical resources for the patient from MongoDB.",
    )


def make_patient_documents_search_tool(patient_id: str) -> StructuredTool:
    def _fn(query: str) -> str:
        """Semantic search over the patient's uploaded document chunks."""
        hits = chromadb_tools.search(
            chromadb_tools.PATIENT_DOCUMENTS,
            query,
            k=settings.RAG_TOP_K,
            where={"patientId": patient_id},
        )
        return _format_hits(hits)

    return StructuredTool.from_function(
        _fn,
        name="chromadb_patient_documents_search",
        description="Semantic search across this patient's uploaded records.",
    )


def make_guidelines_search_tool() -> StructuredTool:
    def _fn(query: str) -> str:
        """Search curated clinical guidelines for relevant guidance."""
        hits = chromadb_tools.search(
            chromadb_tools.CLINICAL_GUIDELINES, query, k=settings.RAG_TOP_K
        )
        return _format_hits(hits)

    return StructuredTool.from_function(
        _fn,
        name="chromadb_guidelines_search",
        description="Search curated clinical guidelines.",
    )


def make_similar_cases_search_tool() -> StructuredTool:
    def _fn(query: str) -> str:
        """Search historical/synthetic similar cases."""
        hits = chromadb_tools.search(
            chromadb_tools.SIMILAR_CASES, query, k=settings.RAG_TOP_K
        )
        return _format_hits(hits)

    return StructuredTool.from_function(
        _fn,
        name="chromadb_similar_cases_search",
        description="Search historical or synthetic similar patient cases.",
    )


def make_web_clinical_search_tool() -> StructuredTool:
    def _fn(query: str, max_results: int = 5) -> str:
        """Search the web for clinical information: lab tests, imaging, differential
        diagnoses, specialist topics, and similar symptom presentations."""
        return web_search_tools.web_clinical_search(query, max_results=max_results)

    return StructuredTool.from_function(
        _fn,
        name="web_clinical_search",
        description=(
            "Search the web for clinical references: lab tests, imaging, "
            "differential diagnoses, and specialist guidance."
        ),
    )

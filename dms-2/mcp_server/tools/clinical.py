"""Clinical helper tools: lab-trend analysis, drug interactions, timeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp_server.tools import _llm, web_search
from rag import vectorstore


def _gather_context(query: str, k: int = 6) -> str:
    hits = vectorstore.search(query, k=k)
    if not hits:
        return ""
    return "\n\n".join(
        f"[source: {h.source} | chunk {h.chunk_index} | sim {h.similarity:.2f}]\n{h.content}"
        for h in hits
    )


def analyze_lab_trends(query: str = "laboratory results over time") -> dict:
    """Detect negative or notable trends across longitudinal lab values."""
    context = _gather_context(query, k=8)
    if not context:
        return {"trends": "", "note": "No indexed documents to analyze."}
    system = (
        "You are a clinical data analyst. From the provided synthetic chart "
        "excerpts, extract laboratory values with their dates and identify "
        "trends (improving, worsening, stable). Cite the source for each trend. "
        "If insufficient data, say so. Never invent values."
    )
    user = f"Question: {query}\n\nChart excerpts:\n{context}"
    return {"trends": _llm.chat(system, user, temperature=0.1)}


def check_drug_interactions(medications: list[str]) -> dict:
    """Flag potential interactions among a list of medications.

    Uses web search when available for current references, otherwise relies on
    the model's general knowledge with an explicit caveat.
    """
    meds = [m for m in (medications or []) if m and m.strip()]
    if not meds:
        return {"interactions": "", "note": "No medications supplied."}
    web = web_search.web_search(
        f"drug interactions between {', '.join(meds)}", max_results=3
    )
    web_block = ""
    if web.get("available"):
        web_block = "\n\nReference snippets:\n" + "\n".join(
            f"- {r.get('title')}: {r.get('content')}" for r in web.get("results", [])
        )
    system = (
        "You are a clinical pharmacist reviewing a synthetic medication list. "
        "Identify clinically significant potential interactions, the mechanism, "
        "and a monitoring suggestion. Be explicit about uncertainty. This is "
        "synthetic data, not medical advice."
    )
    user = f"Medications: {', '.join(meds)}{web_block}"
    return {
        "interactions": _llm.chat(system, user, temperature=0.1),
        "web_search_used": web.get("available", False),
    }


def build_patient_timeline(query: str = "all clinical events") -> dict:
    """Order fragmented events chronologically from indexed documents."""
    context = _gather_context(query, k=10)
    if not context:
        return {"timeline": "", "note": "No indexed documents to order."}
    system = (
        "You reconstruct a chronological clinical timeline from fragmented "
        "synthetic records. Output dated events oldest-to-newest as a bulleted "
        "list (date - event - source). Only use documented dates; flag undated items."
    )
    user = f"Build a timeline for: {query}\n\nExcerpts:\n{context}"
    return {"timeline": _llm.chat(system, user, temperature=0.1)}

"""Curated-knowledge store for the 5.2 Knowledge Curator Agent.

Plain functions (no MCP) that let the Knowledge Curator commit approved,
de-identified, generalizable knowledge once Step-5 consensus is reached:

- ``store_knowledge`` writes the record to the MongoDB ``curated_knowledge``
  collection (the system of record) AND indexes its text into the
  ``agent_learnings`` ChromaDB collection so future runs can semantically
  recall it. This is a direct Chroma write - it deliberately bypasses the
  long-term-memory ``WRITER_AGENTS`` gate, which is reserved for the Guidelines
  and Case Investigator agents.
- ``find_similar_knowledge`` backs the Curator's novelty check by searching the
  existing curated learnings.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import chromadb_tools, mongodb_tools

CURATOR_AGENT = "5.2 Knowledge Curator Agent"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_similar_knowledge(text: str, k: int = 5) -> list[dict]:
    """Return the most similar existing curated learnings for novelty scoring."""
    if not text or not text.strip():
        return []
    hits = chromadb_tools.search(chromadb_tools.AGENT_LEARNINGS, text, k=k)
    return [h.to_dict() for h in hits]


def store_knowledge(record: dict) -> dict:
    """Commit an approved knowledge record.

    ``record`` is expected to contain at least a ``text`` summary plus optional
    ``patientId``, ``runId``, ``title``, ``tags``, ``supportingAgents``, and
    ``noveltyScore``. Returns ``{mongo_id, indexed}``.
    """
    text = (record.get("text") or "").strip()
    if not text:
        return {"mongo_id": None, "indexed": 0, "error": "empty knowledge text"}

    stored = {
        "text": text,
        "title": record.get("title", ""),
        "tags": record.get("tags", []),
        "patientId": record.get("patientId"),
        "runId": record.get("runId"),
        "supportingAgents": record.get("supportingAgents", []),
        "noveltyScore": record.get("noveltyScore"),
        "source": "5.2 Knowledge Curator Agent",
        "curatedAt": _now(),
    }
    mongo_id = mongodb_tools.insert_curated_knowledge(stored)

    metadata = {
        "kind": "curated_knowledge",
        "agent": CURATOR_AGENT,
        "mongoId": mongo_id,
        "title": stored["title"],
        "patientId": stored["patientId"] or "",
        "runId": stored["runId"] or "",
    }
    doc_id = f"curated::{mongo_id}"
    indexed = chromadb_tools.add_documents(
        chromadb_tools.AGENT_LEARNINGS,
        [doc_id],
        [text],
        [metadata],
    )
    return {"mongo_id": mongo_id, "indexed": indexed}

"""Cross-agent long-term memory (spec section 9).

Persistent and cross-run. Readable by every agent (injected into each agent's
context via ``read`` / ``recall_context``). Structured lists live in a singleton
MongoDB document; semantic recall is served by the ``agent_learnings`` and
``clinical_guidelines`` ChromaDB collections.

Write access (this pass) is restricted to the Guidelines and Case Investigator
agents; ``base_agent`` enforces the gate, while these functions are the
low-level primitives.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import chromadb_tools, mongodb_tools

_LTM_COLLECTION = "long_term_memory"
_SINGLETON_ID = "global"

CATEGORIES = [
    "reusable_patterns",
    "guideline_summaries",
    "validated_reasoning_paths",
    "recurring_failures",
    "improvement_notes",
]

# Agents permitted to write long-term memory in this pass.
WRITER_AGENTS = {"2.2 Guidelines Agent", "2.4 Case Investigator Agent"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read() -> dict:
    """Return the structured long-term memory document (all categories)."""
    doc = mongodb_tools.collection(_LTM_COLLECTION).find_one({"_id": _SINGLETON_ID})
    base = {c: [] for c in CATEGORIES}
    if doc:
        for c in CATEGORIES:
            base[c] = doc.get(c, []) or []
    return base


def append(category: str, items: list[dict] | list[str], agent_id: str) -> int:
    """Append items to a long-term-memory category. Returns count appended.

    Raises ``PermissionError`` if ``agent_id`` is not an authorized writer.
    """
    if agent_id not in WRITER_AGENTS:
        raise PermissionError(
            f"Agent '{agent_id}' is not authorized to write long-term memory."
        )
    if category not in CATEGORIES:
        raise ValueError(f"Unknown long-term-memory category: {category}")
    if not items:
        return 0
    stamped = []
    for item in items:
        if isinstance(item, str):
            item = {"text": item}
        stamped.append({**item, "writtenBy": agent_id, "writtenAt": _now()})
    mongodb_tools.collection(_LTM_COLLECTION).update_one(
        {"_id": _SINGLETON_ID},
        {"$push": {category: {"$each": stamped}}, "$set": {"updatedAt": _now()}},
        upsert=True,
    )
    return len(stamped)


def index_learning(text: str, metadata: dict, agent_id: str) -> int:
    """Index a reusable learning into the ``agent_learnings`` collection."""
    if agent_id not in WRITER_AGENTS:
        raise PermissionError(
            f"Agent '{agent_id}' is not authorized to write long-term memory."
        )
    doc_id = f"{agent_id}::{abs(hash(text))}"
    return chromadb_tools.add_documents(
        chromadb_tools.AGENT_LEARNINGS,
        [doc_id],
        [text],
        [{**metadata, "agent": agent_id}],
    )


def index_guideline_summary(text: str, metadata: dict, agent_id: str) -> int:
    """Index a guideline summary into the ``clinical_guidelines`` collection."""
    if agent_id not in WRITER_AGENTS:
        raise PermissionError(
            f"Agent '{agent_id}' is not authorized to write long-term memory."
        )
    doc_id = f"summary::{abs(hash(text))}"
    return chromadb_tools.add_documents(
        chromadb_tools.CLINICAL_GUIDELINES,
        [doc_id],
        [text],
        [{**metadata, "kind": "summary", "agent": agent_id}],
    )


def recall_context(query: str, k: int = 3) -> list[dict]:
    """Semantic recall over agent learnings for any agent to read."""
    hits = chromadb_tools.search(chromadb_tools.AGENT_LEARNINGS, query, k=k)
    return [h.to_dict() for h in hits]

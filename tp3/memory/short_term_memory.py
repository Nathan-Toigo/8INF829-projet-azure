"""Short-term memory helpers (the per-run LangGraph state).

Every agent reads the state at the start of its turn and returns ``memory_updates``
the graph merges in. These helpers cover snapshot persistence and a compact
state summary used by the orchestrator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import mongodb_tools

# State keys that hold gathered clinical content (for summaries / gap checks).
CONTENT_KEYS = [
    "timeline",
    "guidelines",
    "risks",
    "similar_cases",
    "hypotheses",
    "evidence",
    "missing_information",
    "investigation_plan",
    "care_plan",
    "clinical_review",
]


def snapshot(state: dict) -> str | None:
    """Persist a snapshot of the current run state to ``memory_snapshots``."""
    payload = {k: v for k, v in state.items() if k != "long_term_context"}
    try:
        return mongodb_tools.insert_memory_snapshot(
            {
                "runId": state.get("run_id"),
                "patientId": state.get("patient_id"),
                "stepCount": state.get("step_count", 0),
                "state": payload,
            }
        )
    except Exception:
        return None


def summarize_state(state: dict) -> str:
    """Compact human/LLM-readable summary of what the run has gathered so far."""
    lines: list[str] = []
    lines.append(f"Patient: {state.get('patient_id', '?')}")
    lines.append(f"Question: {state.get('patient_question', '')}")
    if state.get("intent"):
        lines.append(f"Intent: {state['intent']}")
    for key in CONTENT_KEYS:
        value = state.get(key) or []
        if isinstance(value, list):
            lines.append(f"{key}: {len(value)} item(s)")
        elif value:
            lines.append(f"{key}: present")
    if state.get("missing_information"):
        lines.append(
            "Missing info: " + "; ".join(str(x) for x in state["missing_information"][:5])
        )
    lines.append(f"Agents run: {', '.join(state.get('agents_run', [])) or 'none'}")
    return "\n".join(lines)


def has_content(state: dict, key: str) -> bool:
    value = state.get(key)
    if isinstance(value, list):
        return len(value) > 0
    return bool(value)

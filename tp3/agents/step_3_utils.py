"""Shared Step 3 constants and state helpers."""

from __future__ import annotations

from config import settings

HIGH_CONFIDENCE_THRESHOLD = 0.75
MAX_HYPOTHESIS_EVIDENCE_ROUNDS = 3
MAX_GAP_RETRY = 2
MAX_STEP_3_ATTEMPTS = settings.MAX_AGENT_RETRIES

STEP_3_LIST_KEYS = [
    "investigation_plan",
    "hypotheses",
    "hypothesis_rationale",
    "evidence",
    "contradictions",
    "unsupported_claims",
    "missing_information",
    "critical_gaps",
    "care_plan",
]


def snapshot_step_3(state: dict) -> dict:
    return {
        "attempt": state.get("step_3_attempt", 1),
        "confidence_score": state.get("confidence_score", 0.0),
        "hypotheses": list(state.get("hypotheses") or []),
        "evidence": list(state.get("evidence") or []),
        "contradictions": list(state.get("contradictions") or []),
    }


def reset_step_3_state(state: dict) -> dict:
    """Partial reset for a full Step 3 retry (keeps Step 2 context)."""
    updates: dict = {key: [] for key in STEP_3_LIST_KEYS}
    updates["confidence_score"] = 0.0
    updates["confidence_rationale"] = ""
    updates["hypotheses_sufficient"] = False
    updates["hypothesis_evidence_rounds"] = 0
    updates["gap_validation_retries"] = 0
    updates["gap_validation_rationale"] = ""
    updates["requires_consensus"] = False
    updates["step_3_care_plan_done"] = False
    updates["step_3_investigation_done"] = False
    updates["step_3_restart_requested"] = False
    previous = list(state.get("step_3_previous_attempts") or [])
    snap = snapshot_step_3(state)
    if snap.get("hypotheses") or snap.get("confidence_score"):
        previous.append(snap)
    updates["step_3_previous_attempts"] = previous
    return updates


def model_override_for_state(state: dict) -> str | None:
    if state.get("step_3_use_alt_model") and settings.OPENROUTER_ALT_STRONG_MODEL:
        return settings.OPENROUTER_ALT_STRONG_MODEL
    return None

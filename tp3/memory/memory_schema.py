"""Memory schemas (spec section 9).

``ShortTermMemory`` is the LangGraph state for one care-plan run: fully
accessible and writable by every agent. ``LongTermMemory`` persists across runs,
is readable by every agent, and (this pass) is written only by the Guidelines
and Case Investigator agents.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ShortTermMemory(TypedDict, total=False):
    # Identity / request
    patient_id: str
    patient_question: str
    intent: str
    run_id: str

    # Gathered context
    timeline: list
    source_documents: list
    missing_dates: list
    guidelines: list
    guideline_sources: list
    risks: list
    red_flags: list
    risk_rationale: list
    similar_cases: list
    case_patterns: list

    # Reasoning (populated in later phases)
    hypotheses: list
    hypothesis_rationale: list
    evidence: list
    contradictions: list
    unsupported_claims: list
    missing_information: list
    critical_gaps: list
    investigation_plan: list

    # Output (populated in later phases)
    care_plan: list
    clinical_summary: str
    follow_up_actions: list
    confidence_score: float
    confidence_rationale: str
    requires_consensus: bool
    patient_explanation: str
    patient_feedback: list
    clinical_review: list

    # Long-term memory injected for agents to read.
    long_term_context: list

    # Routing / control
    next_agent: str | None
    needs_orchestrator: bool

    # Audit (appended via reducers).
    agent_trace: Annotated[list[dict], operator.add]
    tool_calls: Annotated[list[dict], operator.add]
    token_ledger: Annotated[list[dict], operator.add]
    errors: Annotated[list[Any], operator.add]

    # Loop protection.
    step_count: int
    agents_run: list


class LongTermMemory(TypedDict, total=False):
    reusable_patterns: list
    guideline_summaries: list
    validated_reasoning_paths: list
    recurring_failures: list
    improvement_notes: list


def empty_short_term(
    patient_id: str, patient_question: str, run_id: str
) -> ShortTermMemory:
    return ShortTermMemory(
        patient_id=patient_id,
        patient_question=patient_question,
        intent="",
        run_id=run_id,
        timeline=[],
        source_documents=[],
        missing_dates=[],
        guidelines=[],
        guideline_sources=[],
        risks=[],
        red_flags=[],
        risk_rationale=[],
        similar_cases=[],
        case_patterns=[],
        hypotheses=[],
        hypothesis_rationale=[],
        evidence=[],
        contradictions=[],
        unsupported_claims=[],
        missing_information=[],
        critical_gaps=[],
        investigation_plan=[],
        care_plan=[],
        clinical_summary="",
        follow_up_actions=[],
        confidence_score=0.0,
        confidence_rationale="",
        requires_consensus=False,
        patient_explanation="",
        patient_feedback=[],
        clinical_review=[],
        long_term_context=[],
        next_agent=None,
        needs_orchestrator=False,
        agent_trace=[],
        tool_calls=[],
        token_ledger=[],
        errors=[],
        step_count=0,
        agents_run=[],
    )

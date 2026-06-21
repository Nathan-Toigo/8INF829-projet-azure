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
    step_3_phase: str
    step_3_attempt: int
    step_3_use_alt_model: bool
    step_3_best_confidence: float
    step_3_restart_requested: bool
    step_3_best_snapshot: dict
    step_3_previous_attempts: list
    step_3_care_plan_done: bool
    step_3_investigation_done: bool
    step_3_complete: bool
    hypothesis_evidence_rounds: int
    gap_validation_retries: int
    hypotheses_sufficient: bool
    gap_validation_rationale: str

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

    # Step 4 - patient adaptation & clinical review outputs.
    patient_key_points: list
    patient_friendly_explanation: str
    patient_recommended_actions: list
    patient_explanation_reading_level: str
    patient_explanation_previous: str
    patient_appropriateness_passed: bool
    patient_appropriateness_score: float
    patient_appropriateness_issues: list
    patient_appropriateness_suggestions: list
    clinical_review_passed: bool
    clinical_score: float
    clinical_review_assessment: str
    clinical_review_inconsistencies: list
    clinical_review_unsupported_claims: list
    clinical_review_missing_safety_points: list
    clinical_review_attempts: int

    # Step 5 - learning layer (Compliance / Reflection / Knowledge Curator).
    # 5.1 Compliance PII Agent
    sanitized_case: dict
    reidentification_risk: list
    reidentification_risk_score: float
    compliance_notes: list
    compliance_approved: bool
    compliance_reviewed: bool
    # 5.3 Reflection Agent
    agent_scores: list
    reflection_average: float
    reflection_threshold: float
    reflection_flagged: list
    reflection_findings: list
    reflection_approved: bool
    reflection_done: bool
    reflection_evaluation_id: str
    # 5.2 Knowledge Curator Agent
    knowledge_candidate: dict
    knowledge_novelty_score: float
    curator_approved: bool
    knowledge_committed: bool
    knowledge_decision_reason: str
    knowledge_curation_done: bool
    curated_knowledge_id: str

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
        step_3_phase="",
        step_3_attempt=1,
        step_3_use_alt_model=False,
        step_3_best_confidence=0.0,
        step_3_restart_requested=False,
        step_3_best_snapshot={},
        step_3_previous_attempts=[],
        step_3_care_plan_done=False,
        step_3_investigation_done=False,
        step_3_complete=False,
        hypothesis_evidence_rounds=0,
        gap_validation_retries=0,
        hypotheses_sufficient=False,
        gap_validation_rationale="",
        care_plan=[],
        clinical_summary="",
        follow_up_actions=[],
        confidence_score=0.0,
        confidence_rationale="",
        requires_consensus=False,
        patient_explanation="",
        patient_feedback=[],
        clinical_review=[],
        patient_key_points=[],
        patient_friendly_explanation="",
        patient_recommended_actions=[],
        patient_explanation_reading_level="",
        patient_explanation_previous="",
        patient_appropriateness_passed=False,
        patient_appropriateness_score=0.0,
        patient_appropriateness_issues=[],
        patient_appropriateness_suggestions=[],
        clinical_review_passed=False,
        clinical_score=0.0,
        clinical_review_assessment="",
        clinical_review_inconsistencies=[],
        clinical_review_unsupported_claims=[],
        clinical_review_missing_safety_points=[],
        clinical_review_attempts=0,
        sanitized_case={},
        reidentification_risk=[],
        reidentification_risk_score=0.0,
        compliance_notes=[],
        compliance_approved=False,
        compliance_reviewed=False,
        agent_scores=[],
        reflection_average=0.0,
        reflection_threshold=0.0,
        reflection_flagged=[],
        reflection_findings=[],
        reflection_approved=False,
        reflection_done=False,
        reflection_evaluation_id="",
        knowledge_candidate={},
        knowledge_novelty_score=0.0,
        curator_approved=False,
        knowledge_committed=False,
        knowledge_decision_reason="",
        knowledge_curation_done=False,
        curated_knowledge_id="",
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

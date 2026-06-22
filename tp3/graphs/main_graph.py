"""LangGraph main graph: orchestrator supervisor + autonomous agent routing.

One ``StateGraph`` over ``ShortTermMemory``. The orchestrator is the entry point
and sole guaranteed router; agents may hand off directly to a peer or defer back
to the orchestrator. Loop protection (spec section 13) caps total steps so the
workflow always terminates.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from agents import (
    step_1_orchestrator,
    step_2_case_investigator,
    step_2_guidelines,
    step_2_risk,
    step_2_timeline,
    step_3_care_plan,
    step_3_confidence,
    step_3_consensus,
    step_3_evidence,
    step_3_gap_validation,
    step_3_hypothesis,
    step_3_investigation,
    step_4_clinical_review,
    step_4_patient_explanation,
    step_4_patient_representative,
    step_5_compliance,
    step_5_knowledge_curator,
    step_5_reflection,
)
from config import settings
from memory.memory_schema import ShortTermMemory, empty_short_term
from tools import mongodb_tools

ORCHESTRATOR = step_1_orchestrator.AGENT_ID

# All agent nodes registered in the graph (foundation + deferred stubs).
AGENT_MODULES = [
    step_2_timeline,
    step_2_guidelines,
    step_2_risk,
    step_2_case_investigator,
    step_3_investigation,
    step_3_hypothesis,
    step_3_evidence,
    step_3_gap_validation,
    step_3_care_plan,
    step_3_confidence,
    step_3_consensus,
    step_4_patient_explanation,
    step_4_patient_representative,
    step_4_clinical_review,
    step_5_compliance,
    step_5_knowledge_curator,
    step_5_reflection,
]
AGENT_NODES = {m.AGENT_ID: m.run for m in AGENT_MODULES}

_compiled = None


def _wrap(agent_callable, agent_id: str):
    @traceable(
        name=agent_id,
        tags=["care-plan", f"agent:{agent_id}", "patient-workflow"],
    )
    def node(state):
        return agent_callable(state)

    return node


def _route_from_orchestrator(state: dict) -> str:
    if state.get("step_count", 0) >= settings.MAX_AGENT_STEPS:
        return END
    nxt = state.get("next_agent")
    if not nxt or nxt == "END":
        return END
    return nxt if nxt in AGENT_NODES else END


def _route_from_agent(state: dict) -> str:
    if state.get("step_count", 0) >= settings.MAX_AGENT_STEPS:
        return END
    if state.get("needs_orchestrator"):
        return ORCHESTRATOR
    nxt = state.get("next_agent")
    if not nxt or nxt == "END":
        return ORCHESTRATOR
    return nxt if nxt in AGENT_NODES else ORCHESTRATOR


def build_graph():
    graph = StateGraph(ShortTermMemory)

    graph.add_node(ORCHESTRATOR, _wrap(step_1_orchestrator.run, ORCHESTRATOR))
    for agent_id, agent_callable in AGENT_NODES.items():
        graph.add_node(agent_id, _wrap(agent_callable, agent_id))

    graph.add_edge(START, ORCHESTRATOR)

    orchestrator_targets = {a: a for a in AGENT_NODES}
    orchestrator_targets[END] = END
    graph.add_conditional_edges(
        ORCHESTRATOR, _route_from_orchestrator, orchestrator_targets
    )

    agent_targets = {a: a for a in AGENT_NODES}
    agent_targets[ORCHESTRATOR] = ORCHESTRATOR
    agent_targets[END] = END
    for agent_id in AGENT_NODES:
        graph.add_conditional_edges(agent_id, _route_from_agent, agent_targets)

    return graph.compile()


def get_compiled():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def run_workflow(patient_id: str, question: str) -> dict:
    """Run the multi-agent workflow for one patient question.

    Returns the final short-term-memory state (with agent trace, token ledger,
    and tool calls), and persists an ``agent_runs`` record to MongoDB.
    """
    settings.configure_langsmith()
    run_id = f"run-{uuid.uuid4().hex[:10]}"
    initial = empty_short_term(patient_id, question, run_id)

    app = get_compiled()
    config = {"recursion_limit": settings.MAX_AGENT_STEPS * 2 + 5}
    final = app.invoke(initial, config=config)

    try:
        mongodb_tools.insert_agent_run(
            {
                "runId": run_id,
                "patientId": patient_id,
                "question": question,
                "intent": final.get("intent", ""),
                "agentsRun": final.get("agents_run", []),
                "agentTrace": final.get("agent_trace", []),
                "agentOutputs": final.get("agent_outputs", []),
                "tokenLedger": final.get("token_ledger", []),
                "toolCalls": final.get("tool_calls", []),
                "stepCount": final.get("step_count", 0),
                "errors": final.get("errors", []),
                # Gathered content so pages can render results without re-running.
                # Ajout AMAL
                "result": {
                    "timeline": final.get("timeline", []),
                    "missing_dates": final.get("missing_dates", []),
                    "source_documents": final.get("source_documents", []),
                    "guidelines": final.get("guidelines", []),
                    "guideline_sources": final.get("guideline_sources", []),
                    "risks": final.get("risks", []),
                    "red_flags": final.get("red_flags", []),
                    "risk_rationale": final.get("risk_rationale", []),
                    "similar_cases": final.get("similar_cases", []),
                    "case_patterns": final.get("case_patterns", []),
                    "patient_explanation": final.get("patient_explanation", ""),
                    "patient_friendly_explanation": final.get("patient_friendly_explanation", ""),
                    "patient_key_points": final.get("patient_key_points", []),
                    "patient_recommended_actions": final.get("patient_recommended_actions", []),
                    "patient_explanation_reading_level": final.get("patient_explanation_reading_level", ""),
                    "patient_appropriateness_passed": final.get("patient_appropriateness_passed"),
                    "patient_appropriateness_score": final.get("patient_appropriateness_score"),
                    "patient_appropriateness_issues": final.get("patient_appropriateness_issues", []),
                    "clinical_review_passed": final.get("clinical_review_passed"),
                    "clinical_score": final.get("clinical_score"),
                    "clinical_review_assessment": final.get("clinical_review_assessment", ""),
                    "clinical_review_missing_safety_points": final.get("clinical_review_missing_safety_points", []),
                    "clinical_review_unsupported_claims": final.get("clinical_review_unsupported_claims", []),
                    "clinical_review_inconsistencies": final.get("clinical_review_inconsistencies", []),
                    "investigation_plan": final.get("investigation_plan", []),
                    "hypotheses": final.get("hypotheses", []),
                    "hypothesis_rationale": final.get("hypothesis_rationale", []),
                    "evidence": final.get("evidence", []),
                    "contradictions": final.get("contradictions", []),
                    "unsupported_claims": final.get("unsupported_claims", []),
                    "missing_information": final.get("missing_information", []),
                    "critical_gaps": final.get("critical_gaps", []),
                    "care_plan": final.get("care_plan", []),
                    "confidence_score": final.get("confidence_score"),
                    "confidence_rationale": final.get("confidence_rationale", ""),
                    "requires_consensus": final.get("requires_consensus"),
                    "hypotheses_sufficient": final.get("hypotheses_sufficient"),
                    "step_3_attempt": final.get("step_3_attempt"),
                    "step_3_best_confidence": final.get("step_3_best_confidence"),
                },
            }
        )
    except Exception:
        pass

    return final

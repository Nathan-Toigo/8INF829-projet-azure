"""3.4 Gap Validation Agent - decide if hypotheses are sufficient."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents import clinical_tools
from agents.base_agent import AgentResponse, BaseAgent
from agents.step_3_utils import MAX_GAP_RETRY, model_override_for_state
from core import llm

AGENT_ID = "3.4 Gap Validation Agent"
LIKELY_NEXT = [
    "3.2 Hypothesis Agent",
    "3.6 Confidence Assessment Agent",
    "1.1 Clinical Agent Orchestrator",
]


class GapValidationOutput(BaseModel):
    missing_information: list[str] = Field(default_factory=list)
    critical_gaps: list[str] = Field(default_factory=list)
    hypotheses_sufficient: bool = Field(
        default=False,
        description="True when hypotheses are sufficient to answer the question.",
    )
    gap_validation_rationale: str = ""
    next_agent: str = Field(default="3.6 Confidence Assessment Agent")
    handoff_reason: str = ""
    needs_orchestrator: bool = False


_SYSTEM = (
    "You are the 3.4 Gap Validation Agent in an autonomous clinical multi-agent "
    "system. After the hypothesis-evidence cycle, decide whether the current "
    "hypotheses are sufficient to answer the patient's question. Identify any "
    "missing information or critical gaps that would block a confident answer. "
    "Set hypotheses_sufficient=true only when the reasoning is adequate given "
    "available data. Do not invent patient facts. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}. If hypotheses are "
    "insufficient, route back to Hypothesis; if sufficient, route to Confidence. "
    "If blocked, set needs_orchestrator=true and next_agent=null."
)


class GapValidationAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "strong"
    CAN_WRITE_LTM = False

    def execute(self, state):
        if not state.get("hypotheses"):
            return (
                AgentResponse(
                    agent_id=AGENT_ID,
                    status="blocked",
                    memory_updates={},
                    next_agent=None,
                    handoff_reason="No hypotheses available; deferring to orchestrator.",
                    needs_orchestrator=True,
                ),
                [],
                [],
            )

        patient_id = state["patient_id"]
        tools = [
            clinical_tools.make_patient_records_tool(patient_id),
            clinical_tools.make_patient_documents_search_tool(patient_id),
        ]
        retries = state.get("gap_validation_retries", 0)
        user = (
            f"{self.context_block(state)}\n\n"
            f"Gap validation retry: {retries}\n"
            "Assess whether hypotheses are sufficient now."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=tools,
            schema=GapValidationOutput,
            tier=self.TIER,
            model_override=model_override_for_state(state),
        )

        retries += 1
        sufficient = parsed.hypotheses_sufficient
        force_confidence = retries >= MAX_GAP_RETRY and not sufficient

        if sufficient or force_confidence:
            next_agent = "3.6 Confidence Assessment Agent"
            needs_orchestrator = False
            if force_confidence:
                sufficient = True
                reason = (
                    parsed.handoff_reason
                    or f"Max gap retries ({MAX_GAP_RETRY}) reached; proceeding to confidence."
                )
            else:
                reason = parsed.handoff_reason or "Hypotheses sufficient; assessing confidence."
        else:
            next_agent = "3.2 Hypothesis Agent"
            needs_orchestrator = False
            reason = (
                parsed.handoff_reason
                or "Hypotheses insufficient; restarting hypothesis-evidence cycle."
            )

        memory_updates = {
            "missing_information": parsed.missing_information,
            "critical_gaps": parsed.critical_gaps,
            "hypotheses_sufficient": sufficient,
            "gap_validation_rationale": parsed.gap_validation_rationale,
            "gap_validation_retries": retries,
            "step_3_phase": "gap_validation",
        }
        if not sufficient and not force_confidence:
            memory_updates["hypothesis_evidence_rounds"] = 0
        if force_confidence:
            memory_updates["requires_consensus"] = True

        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed",
            memory_updates=memory_updates,
            next_agent=next_agent,
            handoff_reason=reason,
            needs_orchestrator=needs_orchestrator,
        )
        return response, tool_records, token_records


run = GapValidationAgent()

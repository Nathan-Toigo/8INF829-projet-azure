"""3.2 Hypothesis Agent - generate diagnostic hypotheses from ST and LT."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents import clinical_tools
from agents.base_agent import AgentResponse, BaseAgent
from agents.step_3_utils import MAX_HYPOTHESIS_EVIDENCE_ROUNDS, model_override_for_state
from core import llm

AGENT_ID = "3.2 Hypothesis Agent"
LIKELY_NEXT = [
    "3.3 Evidence Validation Agent",
    "3.4 Gap Validation Agent",
    "1.1 Clinical Agent Orchestrator",
]


class Hypothesis(BaseModel):
    label: str = ""
    likelihood: float = Field(default=0.5, description="0-1 likelihood.")
    supporting_factors: list[str] = Field(default_factory=list)
    competing_diagnoses: list[str] = Field(default_factory=list)


class HypothesisOutput(BaseModel):
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    hypothesis_rationale: list[str] = Field(default_factory=list)
    ready_for_gap_validation: bool = Field(
        default=False,
        description="True when no further evidence exchange is needed before gap validation.",
    )
    next_agent: str = Field(default="3.3 Evidence Validation Agent")
    handoff_reason: str = ""
    needs_orchestrator: bool = False


_SYSTEM = (
    "You are the 3.2 Hypothesis Agent in an autonomous clinical multi-agent "
    "system. Generate or revise diagnostic/explanatory hypotheses using Step 2 "
    "context, web investigation findings, long-term memory patterns, and any "
    "evidence or gap feedback already in shared memory. Rank hypotheses by "
    "likelihood. Do not invent patient facts. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}. Route to Evidence "
    "Validation unless the hypothesis-evidence cycle is complete "
    "(ready_for_gap_validation=true, then Gap Validation). If blocked, set "
    "needs_orchestrator=true and next_agent=null."
)


class HypothesisAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "strong"
    CAN_WRITE_LTM = False

    def long_term_query(self, state: dict) -> str:
        return (
            f"{state.get('patient_question', '')} "
            f"{state.get('intent', '')} diagnostic patterns"
        )

    def execute(self, state):
        if not state.get("step_3_investigation_done"):
            return (
                AgentResponse(
                    agent_id=AGENT_ID,
                    status="blocked",
                    memory_updates={},
                    next_agent=None,
                    handoff_reason="Investigation not complete; deferring to orchestrator.",
                    needs_orchestrator=True,
                ),
                [],
                [],
            )

        patient_id = state["patient_id"]
        tools = [
            clinical_tools.make_patient_records_tool(patient_id),
            clinical_tools.make_patient_documents_search_tool(patient_id),
            clinical_tools.make_similar_cases_search_tool(),
            clinical_tools.make_guidelines_search_tool(),
        ]
        rounds = state.get("hypothesis_evidence_rounds", 0)
        user = (
            f"{self.context_block(state)}\n\n"
            f"Hypothesis-evidence round: {rounds}\n"
            "Generate or revise hypotheses now."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=tools,
            schema=HypothesisOutput,
            tier=self.TIER,
            model_override=model_override_for_state(state),
        )

        if not parsed.hypotheses:
            response = AgentResponse(
                agent_id=AGENT_ID,
                status="blocked",
                memory_updates={},
                next_agent=None,
                handoff_reason=parsed.handoff_reason
                or "No hypotheses produced; deferring to orchestrator.",
                needs_orchestrator=True,
            )
            return response, tool_records, token_records

        if parsed.ready_for_gap_validation or rounds >= MAX_HYPOTHESIS_EVIDENCE_ROUNDS:
            next_agent = "3.4 Gap Validation Agent"
            reason = parsed.handoff_reason or "Hypotheses ready for gap validation."
        else:
            next_agent = "3.3 Evidence Validation Agent"
            reason = parsed.handoff_reason or "Hypotheses drafted; validating evidence."

        memory_updates = {
            "hypotheses": [h.model_dump() for h in parsed.hypotheses],
            "hypothesis_rationale": parsed.hypothesis_rationale,
            "step_3_phase": "reasoning",
        }
        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed",
            memory_updates=memory_updates,
            next_agent=next_agent,
            handoff_reason=reason,
            needs_orchestrator=False,
        )
        return response, tool_records, token_records


run = HypothesisAgent()

"""3.3 Evidence Validation Agent - challenge and validate hypotheses."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents import clinical_tools
from agents.base_agent import AgentResponse, BaseAgent
from agents.step_3_utils import MAX_HYPOTHESIS_EVIDENCE_ROUNDS, model_override_for_state
from core import llm

AGENT_ID = "3.3 Evidence Validation Agent"
LIKELY_NEXT = [
    "3.2 Hypothesis Agent",
    "3.4 Gap Validation Agent",
    "1.1 Clinical Agent Orchestrator",
]


class EvidenceItem(BaseModel):
    source: str = ""
    topic: str = ""
    finding: str = ""
    supports_hypothesis: str = ""
    relevance: float = Field(default=0.5, description="0-1 relevance.")


class RevisedHypothesis(BaseModel):
    label: str = ""
    likelihood: float = Field(default=0.5, description="0-1 revised likelihood.")


class EvidenceOutput(BaseModel):
    evidence: list[EvidenceItem] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    revised_hypotheses: list[RevisedHypothesis] = Field(default_factory=list)
    needs_hypothesis_revision: bool = Field(
        default=True,
        description="True if another hypothesis pass is warranted.",
    )
    next_agent: str = Field(default="3.2 Hypothesis Agent")
    handoff_reason: str = ""
    needs_orchestrator: bool = False


_SYSTEM = (
    "You are the 3.3 Evidence Validation Agent in an autonomous clinical "
    "multi-agent system. Analyze, challenge, and validate the current "
    "hypotheses against all available evidence: timeline, guidelines, risks, "
    "similar cases, patient documents, and web investigation findings. Identify "
    "supporting evidence, contradictions, and unsupported claims. Revise "
    "hypothesis likelihoods when warranted. Set needs_hypothesis_revision=false "
    "only when hypotheses are well-supported and ready for gap validation. "
    "Do not invent patient facts. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}. If blocked, set "
    "needs_orchestrator=true and next_agent=null."
)


class EvidenceAgent(BaseAgent):
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
                    handoff_reason="No hypotheses to validate; deferring to orchestrator.",
                    needs_orchestrator=True,
                ),
                [],
                [],
            )

        patient_id = state["patient_id"]
        tools = [
            clinical_tools.make_patient_documents_search_tool(patient_id),
            clinical_tools.make_guidelines_search_tool(),
            clinical_tools.make_similar_cases_search_tool(),
        ]
        rounds = state.get("hypothesis_evidence_rounds", 0)
        user = (
            f"{self.context_block(state)}\n\n"
            f"Hypothesis-evidence round: {rounds}\n"
            "Validate and challenge the current hypotheses now."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=tools,
            schema=EvidenceOutput,
            tier=self.TIER,
            model_override=model_override_for_state(state),
        )

        existing_evidence = list(state.get("evidence") or [])
        new_evidence = [e.model_dump() for e in parsed.evidence]
        hypotheses = list(state.get("hypotheses") or [])
        if parsed.revised_hypotheses:
            revised = {h.label: h.likelihood for h in parsed.revised_hypotheses}
            for hyp in hypotheses:
                label = hyp.get("label", "")
                if label in revised:
                    hyp["likelihood"] = revised[label]

        rounds += 1
        at_limit = rounds >= MAX_HYPOTHESIS_EVIDENCE_ROUNDS
        if parsed.needs_hypothesis_revision and not at_limit:
            next_agent = "3.2 Hypothesis Agent"
            reason = (
                parsed.handoff_reason or "Evidence reviewed; hypothesis revision needed."
            )
        else:
            next_agent = "3.4 Gap Validation Agent"
            reason = parsed.handoff_reason or "Evidence cycle complete; gap validation next."

        memory_updates = {
            "evidence": existing_evidence + new_evidence,
            "contradictions": parsed.contradictions,
            "unsupported_claims": parsed.unsupported_claims,
            "hypotheses": hypotheses,
            "hypothesis_evidence_rounds": rounds,
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


run = EvidenceAgent()

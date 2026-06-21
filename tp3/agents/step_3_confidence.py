"""3.6 Confidence Assessment Agent - score confidence and retry Step 3."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents.base_agent import AgentResponse, BaseAgent
from agents.step_3_utils import (
    HIGH_CONFIDENCE_THRESHOLD,
    MAX_STEP_3_ATTEMPTS,
    model_override_for_state,
    reset_step_3_state,
    snapshot_step_3,
)
from core import llm

AGENT_ID = "3.6 Confidence Assessment Agent"
LIKELY_NEXT = [
    "3.5 Care Planning Agent",
    "4.1 Patient Explanation Agent",
    "1.1 Clinical Agent Orchestrator",
]


class CarePlanItem(BaseModel):
    action: str = ""
    rationale: str = ""
    priority: str = Field(default="medium", description="low | medium | high")


class ConfidenceOutput(BaseModel):
    confidence_score: float = Field(default=0.5, description="0-1 overall confidence.")
    confidence_rationale: str = ""
    care_plan: list[CarePlanItem] = Field(default_factory=list)
    requires_consensus: bool = False
    next_agent: str = Field(default="1.1 Clinical Agent Orchestrator")
    handoff_reason: str = ""
    needs_orchestrator: bool = True


_SYSTEM = (
    "You are the 3.6 Confidence Assessment Agent in an autonomous clinical "
    "multi-agent system. After gap validation confirmed hypotheses are "
    "sufficient, assess overall confidence (0-1) in the reasoning chain. "
    "Explain your score based on evidence quality, gaps, and contradictions. "
    "When confidence is adequate, produce a structured care_plan list of "
    "recommended clinical actions grounded only in shared memory. "
    "Do not invent patient facts. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}. Always return to "
    "the orchestrator after scoring (needs_orchestrator=true, next_agent=null); "
    "the orchestrator will retry Step 3 or proceed to Step 4."
)


class ConfidenceAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "strong"
    CAN_WRITE_LTM = False

    def execute(self, state):
        if not state.get("hypotheses_sufficient"):
            return (
                AgentResponse(
                    agent_id=AGENT_ID,
                    status="blocked",
                    memory_updates={},
                    next_agent=None,
                    handoff_reason="Gap validation not passed; deferring to orchestrator.",
                    needs_orchestrator=True,
                ),
                [],
                [],
            )

        attempt = state.get("step_3_attempt", 1)
        user = (
            f"{self.context_block(state)}\n\n"
            f"Step 3 attempt: {attempt} of {MAX_STEP_3_ATTEMPTS}\n"
            "Assess confidence and produce care plan recommendations if warranted."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=[],
            schema=ConfidenceOutput,
            tier=self.TIER,
            model_override=model_override_for_state(state),
        )

        score = max(0.0, min(1.0, parsed.confidence_score))
        best = max(score, float(state.get("step_3_best_confidence") or 0.0))
        memory_updates: dict = {
            "confidence_score": score,
            "confidence_rationale": parsed.confidence_rationale,
            "step_3_phase": "confidence",
        }
        if parsed.care_plan:
            memory_updates["care_plan"] = [c.model_dump() for c in parsed.care_plan]

        if score >= best:
            memory_updates["step_3_best_confidence"] = score
            memory_updates["step_3_best_snapshot"] = snapshot_step_3(
                {**state, **memory_updates}
            )

        if score >= HIGH_CONFIDENCE_THRESHOLD:
            memory_updates["step_3_complete"] = True
            memory_updates["requires_consensus"] = parsed.requires_consensus
            response = AgentResponse(
                agent_id=AGENT_ID,
                status="completed",
                memory_updates=memory_updates,
                next_agent=None,
                handoff_reason=(
                    parsed.handoff_reason
                    or f"Confidence {score:.2f} meets threshold; Step 3 complete."
                ),
                needs_orchestrator=True,
            )
        elif attempt < MAX_STEP_3_ATTEMPTS:
            reset = reset_step_3_state({**state, **memory_updates})
            reset["step_3_attempt"] = attempt + 1
            reset["step_3_use_alt_model"] = True
            reset["step_3_restart_requested"] = True
            reset["step_3_best_confidence"] = memory_updates.get(
                "step_3_best_confidence", best
            )
            if memory_updates.get("step_3_best_snapshot"):
                reset["step_3_best_snapshot"] = memory_updates["step_3_best_snapshot"]
            memory_updates.update(reset)
            response = AgentResponse(
                agent_id=AGENT_ID,
                status="completed",
                memory_updates=memory_updates,
                next_agent=None,
                handoff_reason=(
                    parsed.handoff_reason
                    or f"Confidence {score:.2f} below {HIGH_CONFIDENCE_THRESHOLD}; "
                    f"restarting Step 3 (attempt {attempt + 1})."
                ),
                needs_orchestrator=True,
            )
        else:
            memory_updates["step_3_complete"] = True
            memory_updates["requires_consensus"] = (
                parsed.requires_consensus or score < HIGH_CONFIDENCE_THRESHOLD
            )
            response = AgentResponse(
                agent_id=AGENT_ID,
                status="completed",
                memory_updates=memory_updates,
                next_agent=None,
                handoff_reason=(
                    parsed.handoff_reason
                    or f"Max Step 3 attempts ({MAX_STEP_3_ATTEMPTS}) reached; "
                    f"best confidence {best:.2f}."
                ),
                needs_orchestrator=True,
            )

        return response, tool_records, token_records


run = ConfidenceAgent()

"""3.5 Care Planning Agent - build the Step 3 action plan."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents import clinical_tools
from agents.base_agent import AgentResponse, BaseAgent
from agents.step_3_utils import model_override_for_state
from core import llm

AGENT_ID = "3.5 Care Planning Agent"
LIKELY_NEXT = [
    "3.1 Investigation Planning Agent",
    "1.1 Clinical Agent Orchestrator",
]


class PlanStep(BaseModel):
    step: str = ""
    goal: str = ""
    priority: str = Field(default="medium", description="low | medium | high")
    suggested_queries: list[str] = Field(default_factory=list)


class CarePlanOutput(BaseModel):
    investigation_plan: list[PlanStep] = Field(default_factory=list)
    follow_up_actions: list[str] = Field(default_factory=list)
    next_agent: str = Field(default="1.1 Clinical Agent Orchestrator")
    handoff_reason: str = ""
    needs_orchestrator: bool = True


_SYSTEM = (
    "You are the 3.5 Care Planning Agent in an autonomous clinical multi-agent "
    "system. Build a structured action plan for Step 3 reasoning based on the "
    "patient question, intent, and all Step 2 context (timeline, guidelines, "
    "risks, similar cases). Each plan step should have a clear goal, priority, "
    "and suggested web/clinical search queries for the Investigation agent. "
    "Do not invent patient facts. Ground the plan only in shared memory. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}. After planning, "
    "return to the orchestrator (needs_orchestrator=true, next_agent=null)."
)


class CarePlanningAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "strong"
    CAN_WRITE_LTM = False

    def execute(self, state):
        tools = [clinical_tools.make_guidelines_search_tool()]
        attempt = state.get("step_3_attempt", 1)
        user = (
            f"{self.context_block(state)}\n\n"
            f"Step 3 attempt: {attempt}\n"
            "Build the Step 3 investigation and reasoning action plan now."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=tools,
            schema=CarePlanOutput,
            tier=self.TIER,
            model_override=model_override_for_state(state),
        )
        memory_updates = {
            "investigation_plan": [s.model_dump() for s in parsed.investigation_plan],
            "step_3_phase": "planning",
            "step_3_care_plan_done": True,
        }
        if parsed.follow_up_actions:
            memory_updates["follow_up_actions"] = parsed.follow_up_actions

        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed" if parsed.investigation_plan else "blocked",
            memory_updates=memory_updates,
            next_agent=None,
            handoff_reason=(
                parsed.handoff_reason
                or "Step 3 action plan stored; returning to orchestrator."
            ),
            needs_orchestrator=True,
        )
        return response, tool_records, token_records


run = CarePlanningAgent()

"""2.3 Risk Agent - identify high-risk issues, red flags, and monitoring needs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents import clinical_tools
from agents.base_agent import AgentResponse, BaseAgent
from core import llm

AGENT_ID = "2.3 Risk Agent"
LIKELY_NEXT = [
    "2.4 Case Investigator Agent",
    "1.1 Clinical Agent Orchestrator",
]


class Risk(BaseModel):
    issue: str = ""
    severity: str = Field(default="moderate", description="low | moderate | high")
    rationale: str = ""
    monitoring: str = Field(default="", description="Recommended monitoring/action.")


class RiskOutput(BaseModel):
    risks: list[Risk] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    risk_rationale: list[str] = Field(default_factory=list)
    next_agent: str = Field(default="2.4 Case Investigator Agent")
    handoff_reason: str = ""
    needs_orchestrator: bool = False


_SYSTEM = (
    "You are the 2.3 Risk Agent in an autonomous clinical multi-agent system. "
    "Your goal is to identify high-risk issues, red flags, and monitoring needs "
    "for this patient. Use the patient records, document search, and the "
    "retrieved guidelines already in memory. Classify severity (low/moderate/"
    "high) and explain your rationale for each risk. Highlight urgent red flags "
    "separately. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}. If blocked, set "
    "needs_orchestrator=true and next_agent=null."
)


class RiskAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "strong"

    def execute(self, state):
        patient_id = state["patient_id"]
        tools = [
            clinical_tools.make_patient_records_tool(patient_id),
            clinical_tools.make_patient_documents_search_tool(patient_id),
            clinical_tools.make_guidelines_search_tool(),
        ]
        user = (
            f"{self.context_block(state)}\n\n"
            "Identify the patient's risks, red flags, and monitoring needs now."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=tools,
            schema=RiskOutput,
            tier=self.TIER,
        )
        memory_updates = {
            "risks": [r.model_dump() for r in parsed.risks],
            "red_flags": parsed.red_flags,
            "risk_rationale": parsed.risk_rationale,
        }
        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed" if parsed.risks else "blocked",
            memory_updates=memory_updates,
            next_agent=None if parsed.needs_orchestrator else parsed.next_agent,
            handoff_reason=parsed.handoff_reason
            or "Risks identified; looking for similar cases next.",
            needs_orchestrator=parsed.needs_orchestrator,
        )
        return response, tool_records, token_records


run = RiskAgent()

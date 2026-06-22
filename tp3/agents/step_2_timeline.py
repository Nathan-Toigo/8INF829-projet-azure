"""2.1 Timeline Agent - build the patient's chronological medical history."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents import clinical_tools
from agents.base_agent import AgentResponse, BaseAgent
from core import llm

AGENT_ID = "2.1 Timeline Agent"
LIKELY_NEXT = [
    "2.2 Guidelines Agent",
    "2.3 Risk Agent",
    "2.4 Case Investigator Agent",
    "1.1 Clinical Agent Orchestrator",
]


class TimelineEvent(BaseModel):
    date: str = Field(default="", description="ISO date (YYYY-MM-DD) or '' if unknown.")
    description: str = ""
    confidence: float = Field(default=0.5, description="0-1 confidence in this event.")
    source: str = Field(default="", description="Source document/file name.")


class TimelineOutput(BaseModel):
    timeline: list[TimelineEvent] = Field(default_factory=list)
    missing_dates: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    next_agent: str = Field(default="2.2 Guidelines Agent")
    handoff_reason: str = ""
    needs_orchestrator: bool = False


_SYSTEM = (
    "You are the 2.1 Timeline Agent in an autonomous clinical multi-agent system. "
    "Your goal is to build the patient's chronological medical history from their "
    "records. Use your tools to read structured resources and search uploaded "
    "documents. Extract dated medical events (diagnoses, labs, imaging, "
    "procedures, medication changes, visits), assign a confidence to each, and "
    "flag events whose dates are missing or uncertain. Work from non real patient data for testing purposes. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}. If you cannot proceed "
    "confidently, set needs_orchestrator=true and next_agent=null."
)


class TimelineAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "small"

    def execute(self, state):
        patient_id = state["patient_id"]
        tools = [
            clinical_tools.make_patient_records_tool(patient_id),
            clinical_tools.make_patient_documents_search_tool(patient_id),
        ]
        user = (
            f"{self.context_block(state)}\n\n"
            "Build the chronological timeline now using the tools."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=tools,
            schema=TimelineOutput,
            tier=self.TIER,
        )
        memory_updates = {
            "timeline": [e.model_dump() for e in parsed.timeline],
            "missing_dates": parsed.missing_dates,
            "source_documents": parsed.source_documents,
        }
        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed" if parsed.timeline else "blocked",
            memory_updates=memory_updates,
            next_agent=None if parsed.needs_orchestrator else parsed.next_agent,
            handoff_reason=parsed.handoff_reason
            or "Timeline built; guideline context is needed next.",
            needs_orchestrator=parsed.needs_orchestrator or not parsed.timeline,
        )
        return response, tool_records, token_records


run = TimelineAgent()

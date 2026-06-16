"""2.2 Guidelines Agent - retrieve relevant clinical guidelines.

Writes long-term memory (guideline summaries) - one of two authorized writers
this pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents import clinical_tools
from agents.base_agent import AgentResponse, BaseAgent
from core import llm
from memory import long_term_memory

AGENT_ID = "2.2 Guidelines Agent"
LIKELY_NEXT = [
    "2.3 Risk Agent",
    "2.4 Case Investigator Agent",
    "1.1 Clinical Agent Orchestrator",
]


class Guideline(BaseModel):
    topic: str = ""
    recommendation: str = ""
    source: str = ""
    relevance: float = Field(default=0.5, description="0-1 relevance to the patient.")


class GuidelinesOutput(BaseModel):
    guidelines: list[Guideline] = Field(default_factory=list)
    guideline_sources: list[str] = Field(default_factory=list)
    guideline_summaries: list[str] = Field(
        default_factory=list,
        description="Concise reusable summaries to persist to long-term memory.",
    )
    next_agent: str = Field(default="2.3 Risk Agent")
    handoff_reason: str = ""
    needs_orchestrator: bool = False


_SYSTEM = (
    "You are the 2.2 Guidelines Agent in an autonomous clinical multi-agent "
    "system. Your goal is to retrieve clinical guidelines relevant to this "
    "patient's conditions and context. Use the guidelines search tool, ranking "
    "results by relevance to the patient's timeline and question. Also produce a "
    "few concise, reusable guideline summaries that will be stored in shared "
    "long-term memory for future runs. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}. If blocked, set "
    "needs_orchestrator=true and next_agent=null."
)


class GuidelinesAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "small"
    CAN_WRITE_LTM = True

    def execute(self, state):
        tools = [clinical_tools.make_guidelines_search_tool()]
        user = (
            f"{self.context_block(state)}\n\n"
            "Retrieve and rank the most relevant clinical guidelines now."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=tools,
            schema=GuidelinesOutput,
            tier=self.TIER,
        )

        # Persist guideline summaries to cross-agent long-term memory.
        persisted = 0
        for summary in parsed.guideline_summaries:
            try:
                long_term_memory.append(
                    "guideline_summaries", [summary], AGENT_ID
                )
                long_term_memory.index_guideline_summary(
                    summary, {"patientId": state.get("patient_id")}, AGENT_ID
                )
                persisted += 1
            except Exception:
                pass

        memory_updates = {
            "guidelines": [g.model_dump() for g in parsed.guidelines],
            "guideline_sources": parsed.guideline_sources,
        }
        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed" if parsed.guidelines else "blocked",
            memory_updates=memory_updates,
            next_agent=None if parsed.needs_orchestrator else parsed.next_agent,
            handoff_reason=(
                parsed.handoff_reason
                or f"Guidelines retrieved ({persisted} summaries persisted to "
                "long-term memory); assessing risks next."
            ),
            needs_orchestrator=parsed.needs_orchestrator,
        )
        return response, tool_records, token_records


run = GuidelinesAgent()

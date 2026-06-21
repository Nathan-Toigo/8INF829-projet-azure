"""3.1 Investigation Planning Agent - web search for clinical investigation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents import clinical_tools
from agents.base_agent import AgentResponse, BaseAgent
from core import llm

AGENT_ID = "3.1 Investigation Planning Agent"
LIKELY_NEXT = [
    "3.2 Hypothesis Agent",
    "1.1 Clinical Agent Orchestrator",
]


class WebFinding(BaseModel):
    source: str = "web"
    topic: str = ""
    finding: str = ""
    relevance: float = Field(default=0.5, description="0-1 relevance.")
    url: str = ""


class InvestigationPlanUpdate(BaseModel):
    step: str = ""
    goal: str = ""
    priority: str = Field(default="medium", description="low | medium | high")
    suggested_queries: list[str] = Field(default_factory=list)
    status: str = Field(
        default="completed",
        description="planned | completed | skipped",
    )
    result_summary: str = Field(
        default="",
        description="Brief outcome of web search for this plan step.",
    )


class InvestigationOutput(BaseModel):
    web_findings: list[WebFinding] = Field(default_factory=list)
    investigation_plan_updates: list[InvestigationPlanUpdate] = Field(
        default_factory=list
    )
    next_agent: str = Field(default="1.1 Clinical Agent Orchestrator")
    handoff_reason: str = ""
    needs_orchestrator: bool = True


_SYSTEM = (
    "You are the 3.1 Investigation Planning Agent in an autonomous clinical "
    "multi-agent system. Execute web searches to find clinical data relevant to "
    "the patient question: similar symptom presentations, recommended lab tests, "
    "imaging, differential diagnoses, and specialist topics. Use the "
    "web_clinical_search tool with queries from the investigation_plan in "
    "shared memory. Summarize findings with source URLs. Do not invent patient "
    "facts. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}. After investigation, "
    "return to the orchestrator (needs_orchestrator=true, next_agent=null)."
)


class InvestigationAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "small"
    CAN_WRITE_LTM = False

    def execute(self, state):
        tools = [
            clinical_tools.make_web_clinical_search_tool(),
            clinical_tools.make_guidelines_search_tool(),
        ]
        user = (
            f"{self.context_block(state)}\n\n"
            "Execute the investigation plan via web search and record findings."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=tools,
            schema=InvestigationOutput,
            tier=self.TIER,
        )

        existing_evidence = list(state.get("evidence") or [])
        new_evidence = [f.model_dump() for f in parsed.web_findings]
        plan = list(state.get("investigation_plan") or [])
        if parsed.investigation_plan_updates:
            plan.extend(u.model_dump() for u in parsed.investigation_plan_updates)

        memory_updates = {
            "evidence": existing_evidence + new_evidence,
            "investigation_plan": plan,
            "step_3_phase": "investigation",
            "step_3_investigation_done": True,
        }
        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed",
            memory_updates=memory_updates,
            next_agent=None,
            handoff_reason=(
                parsed.handoff_reason
                or "Web investigation complete; returning to orchestrator."
            ),
            needs_orchestrator=True,
        )
        return response, tool_records, token_records


run = InvestigationAgent()

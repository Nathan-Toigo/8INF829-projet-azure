"""2.4 Case Investigator Agent - find similar historical/synthetic cases.

Writes long-term memory (reusable patterns / validated reasoning) - one of two
authorized writers this pass.
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

AGENT_ID = "2.4 Case Investigator Agent"
LIKELY_NEXT = [
    "3.2 Hypothesis Agent",
    "3.5 Care Planning Agent",
    "1.1 Clinical Agent Orchestrator",
]


class SimilarCase(BaseModel):
    summary: str = ""
    similarity: float = Field(default=0.5, description="0-1 similarity to this patient.")
    relevant_outcome: str = ""


class CaseOutput(BaseModel):
    similar_cases: list[SimilarCase] = Field(default_factory=list)
    case_patterns: list[str] = Field(default_factory=list)
    reusable_patterns: list[str] = Field(
        default_factory=list,
        description="Generalizable patterns to persist to long-term memory.",
    )
    validated_reasoning_paths: list[str] = Field(default_factory=list)
    next_agent: str = Field(default="1.1 Clinical Agent Orchestrator")
    handoff_reason: str = ""
    needs_orchestrator: bool = True


_SYSTEM = (
    "You are the 2.4 Case Investigator Agent in an autonomous clinical "
    "multi-agent system. Your goal is to find similar historical or synthetic "
    "cases that inform this patient's situation. Use the similar-cases search "
    "tool, rank by similarity, and extract reusable patterns. Persist "
    "generalizable, de-identified patterns and validated reasoning paths to "
    "shared long-term memory for future runs. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}. Prefer returning to "
    "the orchestrator when downstream reasoning agents are not yet available."
)


class CaseInvestigatorAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "small"
    CAN_WRITE_LTM = True

    def execute(self, state):
        tools = [clinical_tools.make_similar_cases_search_tool()]
        user = (
            f"{self.context_block(state)}\n\n"
            "Find and rank similar cases, and extract reusable patterns now."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=tools,
            schema=CaseOutput,
            tier=self.TIER,
        )

        persisted = 0
        patient_id = state.get("patient_id")
        for pattern in parsed.reusable_patterns:
            try:
                long_term_memory.append("reusable_patterns", [pattern], AGENT_ID)
                long_term_memory.index_learning(
                    pattern, {"patientId": patient_id, "kind": "pattern"}, AGENT_ID
                )
                persisted += 1
            except Exception:
                pass
        for path in parsed.validated_reasoning_paths:
            try:
                long_term_memory.append(
                    "validated_reasoning_paths", [path], AGENT_ID
                )
                persisted += 1
            except Exception:
                pass

        memory_updates = {
            "similar_cases": [c.model_dump() for c in parsed.similar_cases],
            "case_patterns": parsed.case_patterns,
        }
        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed",
            memory_updates=memory_updates,
            next_agent=None if parsed.needs_orchestrator else parsed.next_agent,
            handoff_reason=(
                parsed.handoff_reason
                or f"Similar cases reviewed ({persisted} learnings persisted to "
                "long-term memory); returning to orchestrator."
            ),
            needs_orchestrator=parsed.needs_orchestrator,
        )
        return response, tool_records, token_records


run = CaseInvestigatorAgent()

"""4.1 Patient Explanation Agent - produce the final answer for the patient.

Synthesizes the clinical reasoning gathered by foundation/reasoning agents
(timeline, guidelines, risks, similar cases, hypotheses, care plan) into a
clear, plain-language explanation tailored to the patient's question.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents.base_agent import AgentResponse, BaseAgent
from core import llm

AGENT_ID = "4.1 Patient Explanation Agent"
LIKELY_NEXT = [
    "4.2 Patient Representative Agent",
    "4.3 Clinical Review Agent",
    "1.1 Clinical Agent Orchestrator",
]


class PatientExplanationOutput(BaseModel):
    final_answer: str = Field(
        default="",
        description="Plain-language answer to the patient's question.",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="3-6 key takeaways the patient should remember.",
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Concrete next steps (questions to ask, signs to monitor, follow-ups).",
    )
    reading_level: str = Field(
        default="general public",
        description="Estimated reading level of the produced explanation.",
    )
    next_agent: str = Field(default="4.2 Patient Representative Agent")
    handoff_reason: str = ""
    needs_orchestrator: bool = False


_SYSTEM = (
    "You are the 4.1 Patient Explanation Agent in an autonomous clinical "
    "multi-agent system. Your goal is to produce the final answer for the "
    "patient, in clear, plain language, grounded ONLY in the shared memory "
    "produced by upstream agents (timeline, guidelines, risks, similar cases, "
    "and any hypotheses or care plan). Translate medical jargon, keep sentences "
    "short, avoid alarming wording, and never invent facts not present in the "
    "shared memory. Produce a single coherent answer plus a short list of key "
    "points and recommended actions. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}. If blocked, set "
    "needs_orchestrator=true and next_agent=null."
)


class PatientExplanationAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "small"

    def execute(self, state):
        user = (
            f"{self.context_block(state)}\n\n"
            "Write the final patient-facing explanation now, "
            "using only the information already in shared memory."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=[],
            schema=PatientExplanationOutput,
            tier=self.TIER,
        )
        memory_updates = {
            "patient_explanation": parsed.final_answer,
            "patient_key_points": parsed.key_points,
             "patient_friendly_explanation": parsed.final_answer,  # alias pour Care Plan page
            "patient_recommended_actions": parsed.recommended_actions,
            "patient_explanation_reading_level": parsed.reading_level,
        }
        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed" if parsed.final_answer else "blocked",
            memory_updates=memory_updates,
            next_agent=None if parsed.needs_orchestrator else parsed.next_agent,
            handoff_reason=parsed.handoff_reason
            or "Patient-facing explanation drafted; checking appropriateness next.",
            needs_orchestrator=parsed.needs_orchestrator or not parsed.final_answer,
        )
        return response, tool_records, token_records


run = PatientExplanationAgent()
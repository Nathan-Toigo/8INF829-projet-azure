"""1.1 Clinical Agent Orchestrator - supervisor that selects the next agent.

Acts as a supervisor when agents are blocked, uncertain, or missing context. It
validates intent, summarizes the current short-term memory, and routes to the
best next agent. Routing is LLM-proposed but guarded by deterministic heuristics
so the workflow always terminates (loop protection, spec section 13).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents.base_agent import AgentResponse, BaseAgent
from config import settings
from core import llm
from memory import short_term_memory

AGENT_ID = "1.1 Clinical Agent Orchestrator"
END = "END"

STEP_2_ORDER = [
    ("2.1 Timeline Agent", "timeline"),
    ("2.2 Guidelines Agent", "guidelines"),
    ("2.3 Risk Agent", "risks"),
    ("2.4 Case Investigator Agent", "similar_cases"),
]

STEP_3_AGENTS = {
    "3.5 Care Planning Agent",
    "3.1 Investigation Planning Agent",
    "3.2 Hypothesis Agent",
    "3.3 Evidence Validation Agent",
    "3.4 Gap Validation Agent",
    "3.6 Confidence Assessment Agent",
}

STEP_4_ORDER = [
    ("4.1 Patient Explanation Agent", "patient_explanation"),
    ("4.2 Patient Representative Agent", "patient_appropriateness_passed"),
    ("4.3 Clinical Review Agent", "clinical_review_assessment"),
]

FOUNDATION_ORDER = STEP_2_ORDER + STEP_4_ORDER
AVAILABLE_AGENTS = [a for a, _ in FOUNDATION_ORDER] + sorted(STEP_3_AGENTS)


class OrchestratorDecision(BaseModel):
    intent: str = Field(default="", description="Classified intent of the question.")
    next_agent: str = Field(default="", description="Chosen next agent id, or 'END'.")
    reason: str = ""


_SYSTEM = (
    "You are the 1.1 Clinical Agent Orchestrator, the supervisor of an autonomous "
    "clinical multi-agent system. Given the current shared memory, classify the "
    "user's intent and choose the single best next agent to run, or 'END' when "
    "enough has been gathered. Step 3 reasoning is routed deterministically; "
    "prefer the heuristic suggestion for Step 3 agents. "
    f"Choose next_agent strictly from: {', '.join(AVAILABLE_AGENTS)} or END."
)


def _step_2_complete(state: dict) -> bool:
    return all(short_term_memory.has_content(state, key) for _, key in STEP_2_ORDER)


def _step_3_next(state: dict) -> str | None:
    if state.get("step_3_complete"):
        return None

    if state.get("step_3_restart_requested"):
        return "3.5 Care Planning Agent"

    if not state.get("step_3_care_plan_done"):
        return "3.5 Care Planning Agent"
    if not state.get("step_3_investigation_done"):
        return "3.1 Investigation Planning Agent"
    if not short_term_memory.has_content(state, "hypotheses"):
        return "3.2 Hypothesis Agent"
    return None


def _heuristic_next(state: dict) -> str:
    for agent_id, key in STEP_2_ORDER:
        if not short_term_memory.has_content(state, key):
            return agent_id

    if _step_2_complete(state) and not state.get("step_3_complete"):
        step_3 = _step_3_next(state)
        if step_3:
            return step_3

    for agent_id, key in STEP_4_ORDER:
        if not short_term_memory.has_content(state, key):
            return agent_id

    return END


class OrchestratorAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "small"

    def execute(self, state):
        token_records: list = []

        if state.get("step_count", 0) >= settings.MAX_AGENT_STEPS:
            return (
                AgentResponse(
                    agent_id=AGENT_ID,
                    status="completed",
                    memory_updates={},
                    next_agent=END,
                    handoff_reason="Max agent steps reached; ending workflow.",
                    needs_orchestrator=False,
                ),
                [],
                [],
            )

        heuristic = _heuristic_next(state)
        intent = state.get("intent", "")

        try:
            user = (
                f"{self.context_block(state)}\n\n"
                f"Agents already run: {state.get('agents_run', [])}\n"
                f"Heuristic suggestion: {heuristic}\n\n"
                "Classify intent and choose the next agent (or END)."
            )
            decision, usage = llm.invoke_structured(
                step=AGENT_ID,
                system=_SYSTEM,
                user=user,
                schema=OrchestratorDecision,
                tier=self.TIER,
            )
            token_records.append(usage)
            intent = decision.intent or intent
            choice = decision.next_agent.strip()
            reason = decision.reason
        except Exception:
            choice, reason = heuristic, "Heuristic routing (LLM unavailable)."

        if choice not in AVAILABLE_AGENTS and choice != END:
            choice = heuristic
        elif choice != END:
            ran = set(state.get("agents_run", []))
            if choice in ran and choice not in STEP_3_AGENTS:
                choice = heuristic
            elif choice in STEP_3_AGENTS:
                expected = _step_3_next(state)
                if expected and choice != expected:
                    choice = expected

        memory_updates: dict = {}
        if intent and intent != state.get("intent"):
            memory_updates["intent"] = intent
        if state.get("step_3_restart_requested") and choice == "3.5 Care Planning Agent":
            memory_updates["step_3_restart_requested"] = False

        next_agent = None if choice == END else choice
        return (
            AgentResponse(
                agent_id=AGENT_ID,
                status="completed",
                memory_updates=memory_updates,
                next_agent=next_agent,
                handoff_reason=reason or f"Routing to {choice}.",
                needs_orchestrator=False,
            ),
            [],
            token_records,
        )


run = OrchestratorAgent()

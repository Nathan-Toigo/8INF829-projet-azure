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

# Foundation agents available this pass, in default escalation order.
FOUNDATION_ORDER = [
    ("2.1 Timeline Agent", "timeline"),
    ("2.2 Guidelines Agent", "guidelines"),
    ("2.3 Risk Agent", "risks"),
    ("2.4 Case Investigator Agent", "similar_cases"),
]
AVAILABLE_AGENTS = [a for a, _ in FOUNDATION_ORDER]


class OrchestratorDecision(BaseModel):
    intent: str = Field(default="", description="Classified intent of the question.")
    next_agent: str = Field(default="", description="Chosen next agent id, or 'END'.")
    reason: str = ""


_SYSTEM = (
    "You are the 1.1 Clinical Agent Orchestrator, the supervisor of an autonomous "
    "clinical multi-agent system. Given the current shared memory, classify the "
    "user's intent and choose the single best next agent to run, or 'END' when "
    "enough has been gathered. Consider: what information is missing or "
    "unreliable, what output is required next, which agent owns that domain, "
    "whether an agent already ran, and whether the workflow is looping. "
    f"Choose next_agent strictly from: {', '.join(AVAILABLE_AGENTS)} or END."
)


def _heuristic_next(state: dict) -> str:
    """Deterministic fallback / guard: first foundation step with no content."""
    ran = set(state.get("agents_run", []))
    for agent_id, key in FOUNDATION_ORDER:
        if agent_id not in ran and not short_term_memory.has_content(state, key):
            return agent_id
    return END


class OrchestratorAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "small"

    def execute(self, state):
        token_records: list = []

        # Loop protection.
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

        # Ask the LLM supervisor for a routing decision.
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

        # Guard the LLM choice: must be available and not already run; otherwise
        # fall back to the deterministic heuristic to guarantee progress.
        ran = set(state.get("agents_run", []))
        if choice not in AVAILABLE_AGENTS and choice != END:
            choice = heuristic
        elif choice in ran:
            choice = heuristic

        memory_updates = {}
        if intent and intent != state.get("intent"):
            memory_updates["intent"] = intent

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

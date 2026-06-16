"""Standard agent contract + base class (spec section 10).

Every agent returns the same response shape and the graph applies it uniformly:
merge ``memory_updates`` into the short-term state, append an ``agent_trace``
entry, record tool/token usage, and route via ``next_agent`` /
``needs_orchestrator``.

Long-term memory is injected (read) for every agent at the start of its turn.
Write access is gated by ``CAN_WRITE_LTM`` (only Guidelines + Case Investigator
this pass) and enforced again in ``memory.long_term_memory``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from memory import long_term_memory, short_term_memory


class AgentResponse(BaseModel):
    agent_id: str
    status: str = "completed"  # completed | blocked | error
    memory_updates: dict = Field(default_factory=dict)
    next_agent: str | None = None
    handoff_reason: str = ""
    needs_orchestrator: bool = False
    errors: list = Field(default_factory=list)


class BaseAgent:
    AGENT_ID: str = "base"
    TIER: str = "small"
    CAN_WRITE_LTM: bool = False

    def long_term_query(self, state: dict) -> str:
        """Query used to pull relevant long-term learnings for this agent."""
        return state.get("patient_question", "") or state.get("intent", "")

    @staticmethod
    def context_block(state: dict) -> str:
        """Render shared short-term + long-term context for an agent prompt."""
        parts = [short_term_memory.summarize_state(state)]
        ltc = state.get("long_term_context") or []
        if ltc:
            parts.append("\nLong-term memory (reference, read-only):")
            for c in ltc[:8]:
                parts.append(f"- [{c.get('category')}] {str(c.get('item'))[:220]}")
        return "\n".join(parts)

    def execute(self, state: dict) -> tuple[AgentResponse, list, list]:
        """Run the agent. Returns (response, tool_records, token_records)."""
        raise NotImplementedError

    # -- orchestration glue ------------------------------------------------

    def _inject_long_term(self, state: dict) -> list:
        """Read long-term memory so every agent can reference it."""
        try:
            structured = long_term_memory.read()
            recalled = long_term_memory.recall_context(
                self.long_term_query(state), k=3
            )
        except Exception:
            return []
        context: list = []
        for category, items in structured.items():
            for item in (items or [])[-3:]:
                context.append({"category": category, "item": item})
        for hit in recalled:
            context.append({"category": "agent_learnings", "item": hit})
        return context

    def __call__(self, state: dict) -> dict:
        long_term_context = self._inject_long_term(state)
        working = {**state, "long_term_context": long_term_context}
        try:
            response, tool_records, token_records = self.execute(working)
        except Exception as exc:  # pragma: no cover - runtime guard
            response = AgentResponse(
                agent_id=self.AGENT_ID,
                status="error",
                needs_orchestrator=True,
                handoff_reason=f"Unhandled error: {exc}",
                errors=[{"agent": self.AGENT_ID, "error": str(exc)}],
            )
            tool_records, token_records = [], []

        trace_entry = {
            "agent_id": response.agent_id,
            "status": response.status,
            "next_agent": response.next_agent,
            "handoff_reason": response.handoff_reason,
            "needs_orchestrator": response.needs_orchestrator,
            "memory_update_keys": sorted(response.memory_updates.keys()),
            "step": state.get("step_count", 0) + 1,
        }

        update: dict = {}
        update.update(response.memory_updates)
        update["long_term_context"] = long_term_context
        update["next_agent"] = response.next_agent
        update["needs_orchestrator"] = response.needs_orchestrator
        update["agent_trace"] = [trace_entry]
        update["tool_calls"] = tool_records
        update["token_ledger"] = token_records
        update["errors"] = response.errors
        update["step_count"] = state.get("step_count", 0) + 1
        update["agents_run"] = state.get("agents_run", []) + [self.AGENT_ID]

        short_term_memory.snapshot({**working, **update})
        return update

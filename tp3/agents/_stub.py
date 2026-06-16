"""Stub agent factory for deferred (Phase 5-8) agents.

These nodes are wired into the graph for incremental development but defer back
to the orchestrator instead of doing work, so the foundation workflow stays
correct and terminating.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.base_agent import AgentResponse, BaseAgent


def make_stub(agent_id: str):
    class _Stub(BaseAgent):
        AGENT_ID = agent_id

        def execute(self, state):
            return (
                AgentResponse(
                    agent_id=agent_id,
                    status="blocked",
                    memory_updates={},
                    next_agent=None,
                    handoff_reason=(
                        f"{agent_id} is not implemented in this pass; deferring to "
                        "the orchestrator."
                    ),
                    needs_orchestrator=True,
                ),
                [],
                [],
            )

    return _Stub()

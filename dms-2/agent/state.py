"""Typed LangGraph state for the clinical agent."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Identity / request
    patient_id: str
    language: str
    user_request: str

    # Conversation transcript (LangGraph reducer appends).
    messages: Annotated[list, add_messages]

    # Progress
    current_step: str

    # Accumulated audit data (appended via reducers).
    step_results: Annotated[list[dict], operator.add]
    tool_calls: Annotated[list[dict], operator.add]
    token_ledger: Annotated[list[dict], operator.add]
    memory_decisions: Annotated[list[dict], operator.add]

    # Carried between a step and the following memory-evaluation node.
    pending_memory: dict[str, Any]

    # Overwritten snapshots (no reducer => last write wins).
    memory_snapshot: dict
    rag_available: bool
    final_summary: str

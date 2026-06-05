"""Async orchestration helpers bridging the Streamlit UI and the agent graph.

Each top-level user action runs on its own event loop via ``asyncio.run``. MCP
tools use the stateless HTTP transport (a session per invocation), so reloading
the toolset per action keeps things robust across Streamlit reruns while the
``MemoryManager`` persists in session state.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.graph import ClinicalAgent
from agent.mcp_client import load_toolset
from memory.memory_manager import MemoryManager


def run_async(coro):
    return asyncio.run(coro)


async def _make_agent(patient_id: str, language: str, memory: MemoryManager) -> ClinicalAgent:
    toolset = await load_toolset()
    return ClinicalAgent(patient_id, language, toolset, memory)


async def greet_async(patient_id: str, language: str, memory: MemoryManager):
    agent = await _make_agent(patient_id, language, memory)
    return await agent.greet()


async def run_pipeline_async(
    patient_id: str, language: str, memory: MemoryManager, request: str
):
    agent = await _make_agent(patient_id, language, memory)
    return await agent.run(request)


async def followup_async(
    patient_id: str, language: str, memory: MemoryManager, question: str
):
    agent = await _make_agent(patient_id, language, memory)
    return await agent.answer_followup(question)


async def list_tools_async() -> list[str]:
    toolset = await load_toolset()
    return toolset.names()


async def call_tool_async(name: str, args: dict):
    """Invoke a single MCP tool by name (used by the UI for RAG admin tasks)."""
    toolset = await load_toolset()
    tool = toolset.get(name)
    if tool is None:
        raise RuntimeError(f"MCP tool '{name}' unavailable (is the server running?)")
    return await tool.ainvoke(args)

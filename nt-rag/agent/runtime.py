"""Async bridge between the Streamlit UI and the agent."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.chat_agent import ChatAgent
from agent.mcp_client import call_tool_async, list_tools_async


def run_async(coro):
    return asyncio.run(coro)


async def chat_async(agent: ChatAgent, message: str) -> dict:
    return await agent.chat(message)


async def mcp_call_async(name: str, args: dict):
    return await call_tool_async(name, args)


async def list_mcp_tools_async() -> list[str]:
    return await list_tools_async()

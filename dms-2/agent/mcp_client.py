"""MCP client: connects to the standalone HTTP MCP server and adapts its tools.

Uses ``langchain-mcp-adapters`` so the MCP tools become first-class LangChain
tools the agent can invoke (and that LangSmith can trace).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_mcp_adapters.client import MultiServerMCPClient

import config


def tool_result_to_text(result) -> str:
    """Normalize an MCP tool result to plain text.

    langchain-core 1.x returns tool output as a list of content blocks
    (e.g. ``[{"type": "text", "text": "..."}]``); older versions return a plain
    string. This collapses both forms to the underlying text.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for block in result:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    if isinstance(result, dict):
        if "text" in result:
            return str(result["text"])
        return json.dumps(result)
    return str(result)


def tool_result_to_json(result):
    """Best-effort parse of a tool result into Python (or None on failure)."""
    text = tool_result_to_text(result)
    try:
        return json.loads(text)
    except Exception:
        return None


class ClinicalToolset:
    def __init__(self, tools: list):
        self.tools = tools
        self.by_name = {t.name: t for t in tools}

    def get(self, name: str):
        return self.by_name.get(name)

    def names(self) -> list[str]:
        return list(self.by_name)


async def load_toolset() -> ClinicalToolset:
    """Connect to the MCP server and return its tools as LangChain tools."""
    client = MultiServerMCPClient(
        {
            "clinical": {
                "url": config.MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()
    return ClinicalToolset(tools)

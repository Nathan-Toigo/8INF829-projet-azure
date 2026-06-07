"""MCP HTTP client for tools exposed by mcp_server.server."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import config


def tool_result_to_text(result: Any) -> str:
    """Normalize an MCP tool result to plain text."""
    if result is None:
        return ""
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(str(block.text))
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def tool_result_to_json(result: Any) -> Any | None:
    text = tool_result_to_text(result)
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        return None


async def call_tool_async(name: str, args: dict[str, Any]) -> Any:
    """Invoke an MCP tool by name; returns parsed JSON when possible."""
    async with streamablehttp_client(config.MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=args)
            structured = getattr(result, "structuredContent", None)
            if structured is not None:
                return structured
            parsed = tool_result_to_json(result)
            if parsed is not None:
                return parsed
            return tool_result_to_text(result)


async def list_tools_async() -> list[str]:
    """List tool names available on the MCP server."""
    async with streamablehttp_client(config.MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [t.name for t in tools.tools]

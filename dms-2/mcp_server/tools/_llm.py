"""Minimal OpenAI chat helper for LLM-backed MCP tools.

Kept separate from the agent's ``agent/llm.py`` so the MCP server process has no
dependency on the LangGraph/LangChain stack.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI

import config

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.require_openai_key())
    return _client


def chat(system: str, user: str, temperature: float = 0.2) -> str:
    resp = client().chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""

"""Summarize chart text into long-term-memory material."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp_server.tools import _llm

_SYSTEM = (
    "You condense de-identified synthetic clinical text into a compact, factual "
    "summary suitable for long-term patient memory. Keep diagnoses, medications, "
    "key measurements, and dates. Be concise and never invent data."
)


def summarize_history(text: str, focus: str = "") -> str:
    """Summarize a chart section into durable long-term-memory material."""
    if not text.strip():
        return ""
    instruction = f"Focus on: {focus}\n\n" if focus else ""
    user = (
        f"{instruction}Summarize the following clinical text into 4-8 bullet "
        f"points of durable facts:\n\n{text[:12000]}"
    )
    return _llm.chat(_SYSTEM, user, temperature=0.1)

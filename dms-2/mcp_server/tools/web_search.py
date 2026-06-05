"""Web search tool (Tavily) for up-to-date guidelines / drug info.

Degrades gracefully when ``TAVILY_API_KEY`` is unset so the agent can still run
offline; in that case it returns a clear "unavailable" notice rather than raising.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config


def web_search(query: str, max_results: int = 3) -> dict:
    """Search the web for current medical guidelines or drug information."""
    if not config.TAVILY_API_KEY:
        return {
            "available": False,
            "query": query,
            "results": [],
            "note": "Web search unavailable (TAVILY_API_KEY not set). "
            "Proceeding from chart evidence only.",
        }
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        resp = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=True,
        )
        results = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content"),
            }
            for r in resp.get("results", [])
        ]
        return {
            "available": True,
            "query": query,
            "answer": resp.get("answer"),
            "results": results,
        }
    except Exception as exc:  # pragma: no cover - network/runtime guard
        return {
            "available": False,
            "query": query,
            "results": [],
            "note": f"Web search failed: {exc}",
        }

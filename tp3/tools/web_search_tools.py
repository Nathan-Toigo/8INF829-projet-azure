"""Web search for Step 3 clinical investigation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings


def web_clinical_search(query: str, max_results: int = 5) -> str:
    """Search the web for clinical info: lab tests, imaging, differential dx."""
    query = (query or "").strip()
    if not query:
        return "Empty search query."

    if settings.TAVILY_API_KEY:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            response = client.search(query=query, max_results=max_results)
            hits = response.get("results") or []
            if not hits:
                return "No web results found."
            lines = []
            for i, hit in enumerate(hits[:max_results], start=1):
                title = hit.get("title") or "Untitled"
                url = hit.get("url") or ""
                content = (hit.get("content") or "")[:400]
                lines.append(f"[{i}] {title}\nURL: {url}\n{content}")
            return "\n\n".join(lines)
        except Exception as exc:
            return f"Tavily search failed: {exc}"

    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        if not hits:
            return "No web results found."
        lines = []
        for i, hit in enumerate(hits[:max_results], start=1):
            title = hit.get("title") or "Untitled"
            url = hit.get("href") or hit.get("url") or ""
            body = (hit.get("body") or "")[:400]
            lines.append(f"[{i}] {title}\nURL: {url}\n{body}")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"Web search unavailable: {exc}"

"""Standalone HTTP MCP server exposing the clinical toolset.

Run as its own process:

    python -m mcp_server.server

The agent connects to it as an MCP client (see ``agent/mcp_client.py``). Keeping
the tools behind the MCP boundary means RAG, web search, and clinical helpers are
swappable without touching the agent graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

import config
from mcp_server.tools import clinical, rag, summarize, web_search

mcp = FastMCP("clinical-tools", host=config.MCP_HOST, port=config.MCP_PORT)


@mcp.tool()
def rag_search_documents(query: str, k: int = 4) -> list[dict]:
    """Semantic search over uploaded patient records; returns excerpts with provenance."""
    return rag.rag_search_documents(query, k=k)


@mcp.tool()
def ingest_document(
    file_path: str | None = None,
    text: str | None = None,
    source_name: str | None = None,
) -> dict:
    """Add an uploaded file or raw text to the Chroma patient-record index."""
    return rag.ingest_document(file_path=file_path, text=text, source_name=source_name)


@mcp.tool()
def ingest_directory(directory: str) -> dict:
    """Index every supported file in a directory (e.g. the sample patient docs/)."""
    return rag.ingest_directory(directory)


@mcp.tool()
def reset_index() -> dict:
    """Drop and recreate the patient-records collection (fresh index)."""
    return rag.reset_index()


@mcp.tool()
def index_stats() -> dict:
    """Return the indexed chunk count and source files."""
    return rag.index_stats()


@mcp.tool()
def web_search_tool(query: str, max_results: int = 3) -> dict:
    """Search the web (Tavily) for current guidelines / drug info."""
    return web_search.web_search(query, max_results=max_results)


@mcp.tool()
def summarize_history(text: str, focus: str = "") -> str:
    """Summarize chart text into durable long-term-memory material."""
    return summarize.summarize_history(text, focus=focus)


@mcp.tool()
def analyze_lab_trends(query: str = "laboratory results over time") -> dict:
    """Detect negative or notable trends across longitudinal lab values."""
    return clinical.analyze_lab_trends(query)


@mcp.tool()
def check_drug_interactions(medications: list[str]) -> dict:
    """Flag potential interactions among a list of medications."""
    return clinical.check_drug_interactions(medications)


@mcp.tool()
def build_patient_timeline(query: str = "all clinical events") -> dict:
    """Order fragmented clinical events chronologically from indexed documents."""
    return clinical.build_patient_timeline(query)


def main() -> None:
    print(
        f"Starting clinical MCP server on http://{config.MCP_HOST}:{config.MCP_PORT}/mcp",
        file=sys.stderr,
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

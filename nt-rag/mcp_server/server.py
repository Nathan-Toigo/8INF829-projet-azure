"""HTTP MCP server for the nt-rag agent.

Run in a separate terminal:

    python -m mcp_server.server
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

import config
from mcp_server.tools import examples, rag

mcp = FastMCP("nt-rag-tools", host=config.MCP_HOST, port=config.MCP_PORT)


@mcp.tool()
def rag_ask(question: str, top_k: int = 5, chunk_method: str = "fixed_chars") -> dict:
    """Answer a question using the full RAG pipeline (Chroma + Ollama)."""
    return rag.rag_ask(question, top_k=top_k, chunk_method=chunk_method)


@mcp.tool()
def rag_search(query: str, top_k: int = 5, chunk_method: str = "fixed_chars") -> list[dict]:
    """Semantic search over the index without generating an answer."""
    return rag.rag_search(query, top_k=top_k, chunk_method=chunk_method)


@mcp.tool()
def ingest_documents(clear: bool = True, chunk_method: str = "fixed_chars") -> dict:
    """Index all docs/ files (chunk, embed, store in Chroma)."""
    return rag.ingest_documents(clear=clear, chunk_method=chunk_method)


@mcp.tool()
def ingest_file(
    file_path: str,
    chunk_method: str = "fixed_chars",
    replace_existing: bool = True,
) -> dict:
    """Index a single file path into the collection for the given chunk method."""
    return rag.ingest_file(
        file_path, chunk_method=chunk_method, replace_existing=replace_existing
    )


@mcp.tool()
def ingest_uploaded_files(
    file_paths: list[str],
    chunk_method: str = "fixed_chars",
    replace_existing: bool = True,
) -> dict:
    """Index multiple uploaded files (paths under uploads/)."""
    return rag.ingest_uploaded_files(
        file_paths,
        chunk_method=chunk_method,
        replace_existing=replace_existing,
    )


@mcp.tool()
def list_chunk_methods() -> list[dict]:
    """List available document chunking strategies."""
    return rag.list_chunk_methods()


@mcp.tool()
def index_stats(chunk_method: str = "fixed_chars") -> dict:
    """Return indexed chunk count and source files for a chunk method."""
    return rag.index_stats(chunk_method=chunk_method)


@mcp.tool()
def calculate(expression: str) -> dict:
    """Evaluate a simple arithmetic expression, e.g. '(30+1)*19'."""
    return examples.calculate(expression)


@mcp.tool()
def get_current_time() -> dict:
    """Return the current UTC time."""
    return examples.get_current_time()


@mcp.tool()
def count_words(text: str) -> dict:
    """Count words in a text string."""
    return examples.count_words(text)


@mcp.tool()
def echo_message(message: str) -> dict:
    """Echo a message back (demo tool)."""
    return examples.echo_message(message)


@mcp.tool()
def list_document_sources() -> dict:
    """List PDF/DOCX files available in docs/."""
    return examples.list_document_sources()


def main() -> None:
    print(
        f"nt-rag MCP server at http://{config.MCP_HOST}:{config.MCP_PORT}/mcp",
        file=sys.stderr,
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

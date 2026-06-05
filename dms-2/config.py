"""Shared configuration for the dms-2 clinical agent, loaded from environment.

Mirrors the env + OpenAI key handling pattern from the sibling ``dms/config.py``
project, extended with MCP server, Chroma, memory, and LangSmith settings.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _as_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# --- Tavily web search (optional) ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- MCP server ---
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", f"http://{MCP_HOST}:{MCP_PORT}/mcp")

# --- RAG / Chroma ---
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / ".chroma")).resolve()
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "patient_records")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# --- Shared synthetic charts ---
DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT.parent / "docs")).resolve()
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", ROOT / "uploads")).resolve()

# --- Memory ---
SHORT_TERM_MAX_TURNS = int(os.getenv("SHORT_TERM_MAX_TURNS", "8"))
CONTEXT_TOKEN_BUDGET = int(os.getenv("CONTEXT_TOKEN_BUDGET", "3000"))
MEMORY_STORE_DIR = Path(os.getenv("MEMORY_STORE_DIR", ROOT / "memory_store")).resolve()

# --- LangSmith ---
LANGCHAIN_TRACING_V2 = _as_bool(os.getenv("LANGCHAIN_TRACING_V2"))
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "dms-2-clinical-agent")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


def configure_langsmith() -> bool:
    """Export the LangSmith env vars LangGraph/LangChain auto-detect.

    Returns True if tracing is enabled and a key is present.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if LANGCHAIN_TRACING_V2 else "false"
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
    if LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    return LANGCHAIN_TRACING_V2 and bool(LANGCHAIN_API_KEY)


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and set your key."
        )
    return OPENAI_API_KEY

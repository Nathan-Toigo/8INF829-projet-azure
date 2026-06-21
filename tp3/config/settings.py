"""Shared configuration for the ThreeTokens care agent, loaded from environment.

A single ``OPENROUTER_API_KEY`` powers chat, vision OCR, and embeddings via the
OpenAI-compatible OpenRouter endpoint. MongoDB and ChromaDB connection details,
agent loop-protection limits, and LangSmith observability settings also live
here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _as_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


# --- OpenRouter (LLM gateway) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_APP_URL = os.getenv("OPENROUTER_APP_URL", "http://localhost:8501")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "ThreeTokens Care Agent")

# Model tiers (spec section 14).
OPENROUTER_STRONG_MODEL = os.getenv("OPENROUTER_STRONG_MODEL", "openai/gpt-4o")
OPENROUTER_ALT_STRONG_MODEL = os.getenv("OPENROUTER_ALT_STRONG_MODEL", "")
OPENROUTER_SMALL_MODEL = os.getenv("OPENROUTER_SMALL_MODEL", "openai/gpt-4o-mini")
OPENROUTER_VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini")
OPENROUTER_EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
)

# --- MongoDB ---
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "threetokens_care")

# --- ChromaDB ---
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "chroma")).resolve()

# --- Shared synthetic charts / uploads ---
DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT.parent / "docs")).resolve()
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", ROOT / "uploads")).resolve()

# --- Retrieval ---
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# --- Agent loop protection (spec section 13) ---
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "20"))
MAX_AGENT_RETRIES = int(os.getenv("MAX_AGENT_RETRIES", "2"))

# --- Step 5 learning layer thresholds ---
# Reflection (5.3): below-threshold agents are flagged. <=0 => use the run's
# average score as the threshold.
REFLECTION_SCORE_THRESHOLD = float(os.getenv("REFLECTION_SCORE_THRESHOLD", "0"))
# Knowledge Curator (5.2): minimum novelty (1 - max similarity) required to
# consider a case worth curating.
KNOWLEDGE_NOVELTY_THRESHOLD = float(os.getenv("KNOWLEDGE_NOVELTY_THRESHOLD", "0.35"))
# Compliance (5.1): max acceptable reidentification risk score for approval.
REIDENTIFICATION_RISK_THRESHOLD = float(
    os.getenv("REIDENTIFICATION_RISK_THRESHOLD", "0.5")
)

# --- Web search (Step 3 investigation) ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- LangSmith observability ---
LANGCHAIN_TRACING_V2 = _as_bool(os.getenv("LANGCHAIN_TRACING_V2"))
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "threetokens-care-agent")
LANGCHAIN_ENDPOINT = os.getenv(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)


def configure_langsmith() -> bool:
    """Export the LangSmith env vars LangGraph/LangChain auto-detect.

    Returns True if tracing is enabled and an API key is present.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if LANGCHAIN_TRACING_V2 else "false"
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
    if LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    return LANGCHAIN_TRACING_V2 and bool(LANGCHAIN_API_KEY)


def langsmith_project_url() -> str:
    """Best-effort URL to the LangSmith project view."""
    return f"https://smith.langchain.com/projects/p?searchModel=%7B%7D&tab=0"


def require_openrouter_key() -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and set "
            "your OpenRouter key."
        )
    return OPENROUTER_API_KEY


def openrouter_default_headers() -> dict[str, str]:
    """Optional attribution headers OpenRouter uses for rankings."""
    return {
        "HTTP-Referer": OPENROUTER_APP_URL,
        "X-Title": OPENROUTER_APP_TITLE,
    }

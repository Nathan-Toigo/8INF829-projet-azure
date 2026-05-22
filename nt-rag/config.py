"""Configuration for Ollama RAG over docs/."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

RAG_ROOT = Path(__file__).resolve().parent
load_dotenv(RAG_ROOT / ".env")

REPO_ROOT = RAG_ROOT.parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", REPO_ROOT / "docs")).resolve()
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", RAG_ROOT / "data" / "chroma")).resolve()
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", RAG_ROOT / "results")).resolve()
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "docs_rag")

# Ollama (Docker: http://localhost:11434)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", OLLAMA_CHAT_MODEL)
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
OLLAMA_GPU_MODE = os.getenv("OLLAMA_GPU_MODE", "auto")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_FULL_DOC_CHARS = int(os.getenv("MAX_FULL_DOC_CHARS", "120000"))


def normalize_model_for_collection(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", model.replace(":", "-"))


def collection_name_for(chunk_method: str, embed_model: str) -> str:
    norm = normalize_model_for_collection(embed_model)
    return f"docs_rag_{chunk_method}_{norm}"

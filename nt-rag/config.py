"""Configuration for Ollama RAG over docs/."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAG_ROOT = Path(__file__).resolve().parent
load_dotenv(RAG_ROOT / ".env")

REPO_ROOT = RAG_ROOT.parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", REPO_ROOT / "docs")).resolve()
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", RAG_ROOT / "data" / "chroma")).resolve()
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "docs_rag")

# Ollama (Docker: http://localhost:11434)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

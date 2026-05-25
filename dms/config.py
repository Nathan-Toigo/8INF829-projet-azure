"""Shared configuration loaded from environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

DMS_ROOT = Path(__file__).resolve().parent
load_dotenv(DMS_ROOT / ".env")

DOCS_DIR = Path(os.getenv("DOCS_DIR", DMS_ROOT.parent / "docs")).resolve()
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://dms:dms@localhost:5433/dms_rag"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", OPENAI_CHAT_MODEL)
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
# Max characters for full-document prompt injection (approx context guard)
MAX_FULL_DOC_CHARS = int(os.getenv("MAX_FULL_DOC_CHARS", "200000"))
INGEST_ALL_MODELS = os.getenv("INGEST_ALL_MODELS", "0").strip() in ("1", "true", "yes")

# Embedding engines for comparative testing
EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

# OpenAI embedding dimensions (ada-002 and 3-small/large use 1536 for small; large is 3072)
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

CHUNK_METHODS = ["paragraph", "page", "words_250", "llm_optimized"]

TOP_K_PER_METHOD = 5
TOP_CHUNKS_FOR_PROMPT = 2

# Four predefined evaluation questions (synthetic AURALIS patient chart)
TEST_QUESTIONS = [
    (
        "q1",
        "What medications is the patient taking?",
    ),
    (
        "q2",
        "What was the interpretation of the left cervical lymph node fine-needle aspiration pathology and flow cytometry findings from 2021?",
    ),
    (
        "q3",
        "What pulmonary nodule findings were reported on the 2022 CT chest exam and what follow-up surveillance was recommended? "
    ),
    (
        "q4",
        "Quelle surveillance ou suivi a ete recommande pour les nodules pulmonaires ? ",
    ),
]

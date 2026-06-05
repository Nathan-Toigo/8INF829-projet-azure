"""RAG tools: semantic search over patient records + runtime ingestion."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag import ingest, vectorstore


def rag_search_documents(query: str, k: int = 4) -> list[dict]:
    """Semantic search over uploaded patient records (Chroma).

    Returns ranked excerpts with provenance (source file + similarity) so the
    agent can cite where each finding came from.
    """
    hits = vectorstore.search(query, k=k)
    return [h.to_dict() for h in hits]


def ingest_document(
    file_path: str | None = None,
    text: str | None = None,
    source_name: str | None = None,
) -> dict:
    """Add a document to the Chroma index, by file path or raw text."""
    if file_path:
        return ingest.ingest_file(Path(file_path), source_name=source_name)
    if text:
        return ingest.ingest_text(text, source_name=source_name or "pasted_text")
    return {"error": "Provide either file_path or text."}


def ingest_directory(directory: str) -> dict:
    """Index every supported file in a directory (e.g. the sample patient)."""
    results = ingest.ingest_directory(Path(directory))
    return {
        "documents": len(results),
        "chunks_indexed": sum(r["chunks_indexed"] for r in results),
        "files": results,
    }


def reset_index() -> dict:
    """Drop and recreate the patient-records collection (fresh index)."""
    vectorstore.reset_collection()
    return {"status": "reset"}


def index_stats() -> dict:
    """Return chunk count and indexed source files."""
    return vectorstore.stats()

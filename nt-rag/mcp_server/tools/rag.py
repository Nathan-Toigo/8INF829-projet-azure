"""RAG tools wrapping ingest.py, query.py and store.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from chunking import CHUNK_METHODS
from ingest import ingest_files, run_ingest
from ollama_client import embed_texts
from query import ask_with_metrics, format_context
from store import get_collection, query_collection

CHUNK_METHOD_LABELS: dict[str, str] = {
    "fixed_chars": "Fixed characters (size/overlap from config)",
    "paragraph": "Split on paragraph breaks",
    "page": "One chunk per PDF page (split if too long)",
    "words_250": "250-word windows with overlap",
}


def collection_for_method(chunk_method: str = "fixed_chars") -> str:
    return config.collection_name_for(chunk_method, config.OLLAMA_EMBED_MODEL)


def _coerce_top_k(top_k: int | str | None) -> int:
    if top_k is None:
        return config.TOP_K
    try:
        return max(1, int(top_k))
    except (TypeError, ValueError):
        return config.TOP_K


def list_chunk_methods() -> list[dict]:
    """Return available chunking strategies."""
    return [
        {"id": m, "label": CHUNK_METHOD_LABELS.get(m, m)}
        for m in CHUNK_METHODS
    ]


def rag_ask(
    question: str,
    top_k: int | None = None,
    chunk_method: str = "fixed_chars",
) -> dict:
    """Full RAG pipeline: retrieve chunks then generate an answer with Ollama."""
    k = _coerce_top_k(top_k)
    coll = collection_for_method(chunk_method)
    result = ask_with_metrics(question, top_k=k, collection_name=coll)
    return {
        "answer": result["answer"],
        "sources": result["retrieved_chunks"],
        "metrics": result["metrics"],
        "chunk_method": chunk_method,
        "collection_name": coll,
    }


def rag_search(
    query: str,
    top_k: int | None = None,
    chunk_method: str = "fixed_chars",
) -> list[dict]:
    """Semantic search without LLM generation; returns relevant excerpts."""
    k = _coerce_top_k(top_k)
    coll = collection_for_method(chunk_method)
    collection = get_collection(collection_name=coll)
    if collection.count() == 0:
        return []

    query_emb = embed_texts([query])[0]
    results = query_collection(collection, query_emb, k)
    docs = results.get("documents") or [[]]
    metas = results.get("metadatas") or [[]]
    dists = results.get("distances") or [[]]

    hits: list[dict] = []
    if not docs or not docs[0]:
        return hits
    for i, text in enumerate(docs[0]):
        meta = metas[0][i] if metas and metas[0] else {}
        hits.append(
            {
                "source_file": meta.get("source_file"),
                "page_index": meta.get("page_index"),
                "distance": dists[0][i] if dists and dists[0] else None,
                "excerpt": text[:800],
            }
        )
    return hits


def ingest_documents(
    clear: bool = True,
    chunk_method: str = "fixed_chars",
) -> dict:
    """Index all PDF/DOCX files from docs/ using the selected chunk method."""
    return run_ingest(clear=clear, chunk_method=chunk_method)


def ingest_file(
    file_path: str,
    chunk_method: str = "fixed_chars",
    replace_existing: bool = True,
) -> dict:
    """Index a single file (append to collection, optionally replace same source)."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return ingest_files(
        [path],
        chunk_method=chunk_method,
        clear=False,
        replace_sources=replace_existing,
    )


def ingest_uploaded_files(
    file_paths: list[str],
    chunk_method: str = "fixed_chars",
    replace_existing: bool = True,
) -> dict:
    """Index multiple uploaded files into the active collection."""
    paths = [Path(p) for p in file_paths]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Files not found: {', '.join(missing)}")
    return ingest_files(
        paths,
        chunk_method=chunk_method,
        clear=False,
        replace_sources=replace_existing,
    )


def index_stats(chunk_method: str = "fixed_chars") -> dict:
    """Return vector count and indexed source files for a chunk-method collection."""
    coll = collection_for_method(chunk_method)
    collection = get_collection(collection_name=coll)
    count = collection.count()
    sources: list[str] = []
    if count > 0:
        sample = collection.get(limit=min(count, 500), include=["metadatas"])
        metas = sample.get("metadatas") or []
        sources = sorted({m.get("source_file", "?") for m in metas if m})
    return {
        "collection_name": coll,
        "chunk_method": chunk_method,
        "chunk_count": count,
        "sources": sources,
        "docs_dir": str(config.DOCS_DIR),
        "uploads_dir": str(config.UPLOADS_DIR),
    }


def rag_context_preview(
    query: str,
    top_k: int = 3,
    chunk_method: str = "fixed_chars",
) -> str:
    """Formatted RAG context preview (debug helper)."""
    k = top_k or config.TOP_K
    coll = collection_for_method(chunk_method)
    collection = get_collection(collection_name=coll)
    if collection.count() == 0:
        return "(empty index - run ingest first)"
    query_emb = embed_texts([query])[0]
    results = query_collection(collection, query_emb, k)
    return format_context(results)

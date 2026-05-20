#!/usr/bin/env python3
"""Ingest documents: chunk with comparative methods, embed, store in pgvector."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collections import defaultdict

from openai import OpenAI

import config
from chunking import TextChunk, chunk_all_methods
from database import clear_chunks, init_schema, insert_chunks
from document_loader import load_all_documents, load_document


def get_openai_client() -> OpenAI:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=config.OPENAI_API_KEY)


def embed_texts(
    client: OpenAI,
    texts: list[str],
    model: str,
) -> list[list[float]]:
    if not texts:
        return []
    kwargs: dict = {"model": model, "input": texts}
    # Matryoshka: 3.x models support reduced dimensions for a unified table
    if model.startswith("text-embedding-3"):
        kwargs["dimensions"] = 1536
    response = client.embeddings.create(**kwargs)
    return [item.embedding for item in response.data]


def embed_batch(
    client: OpenAI,
    chunks: list[TextChunk],
    model: str,
    batch_size: int = 32,
) -> tuple[list[list[float]], dict]:
    embeddings: list[list[float]] = []
    t0 = time.perf_counter()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.content for c in batch]
        embeddings.extend(embed_texts(client, texts, model))
    elapsed = time.perf_counter() - t0
    metrics = {
        "embedding_model": model,
        "chunk_count": len(chunks),
        "elapsed_sec": round(elapsed, 3),
        "chunks_per_sec": round(len(chunks) / elapsed, 2) if elapsed else 0,
    }
    return embeddings, metrics


def build_rows(
    chunks: list[TextChunk],
    embeddings: list[list[float]],
    model: str,
) -> list[dict]:
    rows = []
    for chunk, emb in zip(chunks, embeddings):
        rows.append(
            {
                "source_file": chunk.source_file,
                "chunk_method": chunk.chunk_method,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "word_count": chunk.word_count,
                "content": chunk.content,
                "embedding_model": model,
                "embedding": emb,
                "metadata": {"ingest_version": "1"},
            }
        )
    return rows


def summarize_chunks(all_chunks: dict[str, list[TextChunk]]) -> dict:
    summary: dict = {}
    for method, chunks in all_chunks.items():
        if not chunks:
            summary[method] = {"count": 0}
            continue
        words = [c.word_count for c in chunks]
        summary[method] = {
            "count": len(chunks),
            "avg_words": round(sum(words) / len(words), 1),
            "min_words": min(words),
            "max_words": max(words),
        }
    return summary


def run_ingest(clear: bool = True) -> dict:
    """Run full ingest pipeline; returns metrics dict."""
    client = get_openai_client()
    init_schema()
    if clear:
        clear_chunks()

    #docs = load_all_documents(config.DOCS_DIR)
    docs = [load_document(config.DOCS_DIR / "example_patient_2.pdf")]

    if not docs:
        raise FileNotFoundError(f"No documents found in {config.DOCS_DIR}")

    models = (
        config.EMBEDDING_MODELS
        if config.INGEST_ALL_MODELS
        else [config.OPENAI_EMBEDDING_MODEL]
    )

    method_chunks: dict[str, list[TextChunk]] = defaultdict(list)
    for doc in docs:
        per_doc = chunk_all_methods(doc)
        for method, chunks in per_doc.items():
            method_chunks[method].extend(chunks)

    chunk_summary = summarize_chunks(dict(method_chunks))
    print("\n=== Chunking summary (comparative methods) ===")
    for method, stats in chunk_summary.items():
        print(f"  {method}: {stats}")

    ingest_metrics: dict = {
        "documents_loaded": len(docs),
        "chunk_summary": chunk_summary,
        "embedding_runs": [],
        "total_rows_inserted": 0,
    }

    for model in models:
        print(f"\n=== Embedding with {model} ===")
        all_for_model: list[TextChunk] = []
        for method in config.CHUNK_METHODS:
            all_for_model.extend(method_chunks.get(method, []))

        embeddings, emb_metrics = embed_batch(client, all_for_model, model)
        rows = build_rows(all_for_model, embeddings, model)
        inserted = insert_chunks(rows)
        emb_metrics["rows_inserted"] = inserted
        ingest_metrics["embedding_runs"].append(emb_metrics)
        ingest_metrics["total_rows_inserted"] += inserted
        print(f"  Inserted {inserted} rows in {emb_metrics['elapsed_sec']}s")

    print("\n=== Ingest complete ===")
    print(f"  Documents: {ingest_metrics['documents_loaded']}")
    print(f"  Total rows: {ingest_metrics['total_rows_inserted']}")
    return ingest_metrics


def main() -> int:
    try:
        run_ingest()
        return 0
    except Exception as exc:
        print(f"INGEST ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

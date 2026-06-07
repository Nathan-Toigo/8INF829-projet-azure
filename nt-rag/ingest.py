#!/usr/bin/env python3
"""Ingest docs/: chunk, embed with Ollama, store in Chroma."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from chunking import chunk_documents
from documents import load_all_documents, load_document
from ollama_client import embed_texts
from store import add_chunks, clear_collection, collection_vector_count, delete_by_source, get_collection


def ingest_metrics_for_existing_collection(
    collection_name: str,
    *,
    chunk_method: str,
    embed_model: str,
) -> dict:
    """Metrics snapshot when reusing vectors already stored in Chroma."""
    n = collection_vector_count(collection_name)
    return {
        "documents": None,
        "chunks": n,
        "vectors": n,
        "embed_seconds": 0,
        "ingest_total_sec": 0,
        "chunk_method": chunk_method,
        "embed_model": embed_model,
        "collection_name": collection_name,
        "ingest_reused": True,
    }


def should_skip_ingest(collection_name: str) -> bool:
    return collection_vector_count(collection_name) > 0


def _embed_and_store(
    chunks: list,
    *,
    coll: str,
    emb: str,
    clear: bool,
) -> tuple[list[list[float]], float, int]:
    if clear:
        clear_collection(coll)

    embeddings: list[list[float]] = []
    batch_size = 16
    n_batches = (len(chunks) + batch_size - 1) // batch_size
    t0 = time.perf_counter()
    for bi, i in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[i : i + batch_size]
        texts = [c.content for c in batch]
        embeddings.extend(embed_texts(texts, model=emb))
        if n_batches > 1 and (bi == 1 or bi == n_batches or bi % 4 == 0):
            print(f"    Embed batch {bi}/{n_batches}...", flush=True)
    embed_sec = time.perf_counter() - t0
    print(f"    Embedded with '{emb}' in {embed_sec:.2f}s", flush=True)

    collection = get_collection(collection_name=coll)
    add_chunks(collection, chunks, embeddings)
    return embeddings, embed_sec, collection.count()


def ingest_files(
    file_paths: list[Path],
    *,
    chunk_method: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embed_model: str | None = None,
    collection_name: str | None = None,
    clear: bool = False,
    replace_sources: bool = True,
) -> dict:
    """Chunk, embed and store specific files (append or replace per source)."""
    if not file_paths:
        raise RuntimeError("No files provided for ingestion")

    method = chunk_method or "fixed_chars"
    size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    emb = embed_model or config.OLLAMA_EMBED_MODEL
    coll = collection_name or config.collection_name_for(method, emb)

    docs = [load_document(p) for p in file_paths]
    t_ingest = time.perf_counter()
    chunks = chunk_documents(docs, method=method, chunk_size=size, chunk_overlap=overlap)
    print(
        f"    Loaded {len(docs)} file(s) -> {len(chunks)} chunks ({method})",
        flush=True,
    )

    collection = get_collection(collection_name=coll)
    if clear:
        clear_collection(coll)
        collection = get_collection(collection_name=coll)
    elif replace_sources:
        for doc in docs:
            delete_by_source(collection, doc.source_file)

    _, embed_sec, count = _embed_and_store(
        chunks, coll=coll, emb=emb, clear=False
    )
    ingest_sec = time.perf_counter() - t_ingest
    print(f"    Stored {count} vectors in '{coll}'", flush=True)
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "vectors": count,
        "embed_seconds": round(embed_sec, 2),
        "ingest_total_sec": round(ingest_sec, 2),
        "chunk_method": method,
        "embed_model": emb,
        "collection_name": coll,
        "files": [p.name for p in file_paths],
    }


def run_ingest(
    *,
    clear: bool = True,
    chunk_method: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embed_model: str | None = None,
    collection_name: str | None = None,
) -> dict:
    method = chunk_method or "fixed_chars"
    size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    emb = embed_model or config.OLLAMA_EMBED_MODEL
    coll = collection_name or config.collection_name_for(method, emb)

    docs = load_all_documents(config.DOCS_DIR)
    if not docs:
        raise RuntimeError(f"No PDF/DOCX files found in {config.DOCS_DIR}")

    t_ingest = time.perf_counter()
    chunks = chunk_documents(docs, method=method, chunk_size=size, chunk_overlap=overlap)
    print(f"    Loaded {len(docs)} documents -> {len(chunks)} chunks ({method})", flush=True)
    if chunks:
        lens = [len(c.content) for c in chunks]
        print(
            f"    Chunk sizes (chars): min={min(lens)} max={max(lens)} "
            f"avg={sum(lens) // len(lens)}",
            flush=True,
        )

    if clear:
        clear_collection(coll)

    _, embed_sec, count = _embed_and_store(
        chunks, coll=coll, emb=emb, clear=False
    )
    ingest_sec = time.perf_counter() - t_ingest
    print(f"    Stored {count} vectors in '{coll}'", flush=True)
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "vectors": count,
        "embed_seconds": round(embed_sec, 2),
        "ingest_total_sec": round(ingest_sec, 2),
        "chunk_method": method,
        "embed_model": emb,
        "collection_name": coll,
    }


if __name__ == "__main__":
    run_ingest(clear=True)

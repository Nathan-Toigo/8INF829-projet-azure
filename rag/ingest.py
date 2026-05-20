#!/usr/bin/env python3
"""Ingest docs/: chunk, embed with Ollama, store in Chroma."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from chunking import chunk_documents
from documents import load_all_documents
from ollama_client import embed_texts
from store import add_chunks, clear_collection, get_collection


def run_ingest(*, clear: bool = True) -> dict:
    docs = load_all_documents(config.DOCS_DIR)
    if not docs:
        raise RuntimeError(f"No PDF/DOCX files found in {config.DOCS_DIR}")

    chunks = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")

    if clear:
        clear_collection()

    embeddings: list[list[float]] = []
    batch_size = 16
    t0 = time.perf_counter()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.content for c in batch]
        embeddings.extend(embed_texts(texts))
    embed_sec = time.perf_counter() - t0
    print(
        f"Embedded with Ollama model '{config.OLLAMA_EMBED_MODEL}' "
        f"in {embed_sec:.2f}s"
    )

    collection = get_collection()
    add_chunks(collection, chunks, embeddings)
    count = collection.count()
    print(f"Stored {count} vectors in Chroma ({config.CHROMA_DIR})")
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "vectors": count,
        "embed_seconds": round(embed_sec, 2),
    }


if __name__ == "__main__":
    run_ingest(clear=True)

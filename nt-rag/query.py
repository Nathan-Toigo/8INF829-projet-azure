#!/usr/bin/env python3
"""RAG query: retrieve chunks from Chroma, answer with Ollama chat."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from ollama_client import chat_completion, embed_texts
from store import get_collection, query_collection

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context from medical and clinical documents.
If the context does not contain enough information, say so clearly.
Cite source file names when relevant. Be concise and factual."""


def format_context(query_result: dict) -> str:
    docs = query_result.get("documents") or [[]]
    metas = query_result.get("metadatas") or [[]]
    if not docs or not docs[0]:
        return "(no matching context)"

    blocks: list[str] = []
    for text, meta in zip(docs[0], metas[0]):
        source = meta.get("source_file", "unknown")
        page = meta.get("page_index", "?")
        blocks.append(f"[{source} p.{page}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def ask_with_metrics(
    question: str,
    *,
    top_k: int | None = None,
    collection_name: str | None = None,
    embed_model: str | None = None,
    chat_model: str | None = None,
) -> dict[str, Any]:
    k = top_k or config.TOP_K
    coll = collection_name or config.COLLECTION_NAME
    emb = embed_model or config.OLLAMA_EMBED_MODEL
    chat = chat_model or config.OLLAMA_CHAT_MODEL

    collection = get_collection(collection_name=coll)
    if collection.count() == 0:
        raise RuntimeError(
            f"Vector store '{coll}' is empty. Run ingest for this collection first."
        )

    t0 = time.perf_counter()
    query_emb = embed_texts([question], model=emb)[0]
    query_embed_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    results = query_collection(collection, query_emb, k)
    retrieve_ms = (time.perf_counter() - t1) * 1000

    context = format_context(results)
    metas = results.get("metadatas") or [[]]
    retrieved = []
    if metas and metas[0]:
        docs = results.get("documents") or [[]]
        dists = results.get("distances") or [[]]
        for i, meta in enumerate(metas[0]):
            retrieved.append(
                {
                    "source_file": meta.get("source_file"),
                    "page_index": meta.get("page_index"),
                    "chunk_method": meta.get("chunk_method"),
                    "distance": dists[0][i] if dists and dists[0] else None,
                    "preview": (docs[0][i][:200] + "...") if docs and docs[0] else "",
                }
            )

    t2 = time.perf_counter()
    answer = chat_completion(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer based on the context above."
                ),
            },
        ],
        model=chat,
    )
    chat_ms = (time.perf_counter() - t2) * 1000

    return {
        "question": question,
        "answer": answer,
        "rag_context": context,
        "retrieved_chunks": retrieved,
        "collection_name": coll,
        "embed_model": emb,
        "chat_model": chat,
        "metrics": {
            "query_embed_ms": round(query_embed_ms, 2),
            "retrieve_ms": round(retrieve_ms, 2),
            "chat_ms": round(chat_ms, 2),
            "total_ms": round(query_embed_ms + retrieve_ms + chat_ms, 2),
        },
    }


def ask(question: str, *, top_k: int | None = None) -> str:
    coll = config.collection_name_for("fixed_chars", config.OLLAMA_EMBED_MODEL)
    return ask_with_metrics(question, top_k=top_k, collection_name=coll)["answer"]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Summarize the main clinical themes in these records."
    print(ask(q))

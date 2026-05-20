#!/usr/bin/env python3
"""RAG query: retrieve chunks from Chroma, answer with Ollama chat."""

from __future__ import annotations

import sys
from pathlib import Path

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


def ask(question: str, *, top_k: int | None = None) -> str:
    k = top_k or config.TOP_K
    collection = get_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "Vector store is empty. Run: python run.py ingest"
        )

    query_emb = embed_texts([question])[0]
    results = query_collection(collection, query_emb, k)
    context = format_context(results)

    return chat_completion(
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
        ]
    )


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Summarize the main clinical themes in these records."
    print(ask(q))

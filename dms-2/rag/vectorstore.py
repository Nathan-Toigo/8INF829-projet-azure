"""In-process Chroma vector store for patient records.

A single on-disk persistent collection is shared between the Streamlit UI
process (which ingests uploads) and the standalone MCP server process (which
searches). OpenAI embeddings (``text-embedding-3-small`` by default) are used so
both processes produce comparable vectors.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.utils import embedding_functions

import config

_client: chromadb.api.ClientAPI | None = None
_ef = None


@dataclass
class SearchHit:
    rank: int
    content: str
    source: str
    chunk_index: int
    distance: float
    similarity: float

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "content": self.content,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "distance": round(self.distance, 4),
            "similarity": round(self.similarity, 4),
        }


def _embedding_function():
    global _ef
    if _ef is None:
        config.require_openai_key()
        _ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.OPENAI_API_KEY,
            model_name=config.OPENAI_EMBEDDING_MODEL,
        )
    return _ef


def _get_client() -> chromadb.api.ClientAPI:
    global _client
    if _client is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def get_collection():
    """Return the shared patient-records collection.

    The collection handle is intentionally NOT cached: ``get_or_create_collection``
    is cheap and always resolves the current collection, which avoids stale
    handles after a reset (the cause of "Collection ... does not exist").
    """
    return _get_client().get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """Drop and recreate the collection (a fresh, per-session index)."""
    client = _get_client()
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass
    client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(source: str, chunks: list[str]) -> int:
    """Add text chunks for one source document. Returns count added."""
    if not chunks:
        return 0
    collection = get_collection()
    ids = [f"{source}::chunk::{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": source, "chunk_index": i} for i in range(len(chunks))
    ]
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def search(query: str, k: int | None = None) -> list[SearchHit]:
    """Cosine-similarity search; returns ranked hits with provenance."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    k = k or config.RAG_TOP_K
    res = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    hits: list[SearchHit] = []
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        hits.append(
            SearchHit(
                rank=rank,
                content=doc,
                source=str(meta.get("source", "unknown")),
                chunk_index=int(meta.get("chunk_index", -1)),
                distance=float(dist),
                similarity=1.0 - float(dist),
            )
        )
    return hits


def stats() -> dict:
    collection = get_collection()
    sources: set[str] = set()
    try:
        got = collection.get(include=["metadatas"])
        for meta in got.get("metadatas") or []:
            if meta and "source" in meta:
                sources.add(str(meta["source"]))
    except Exception:
        pass
    return {"chunk_count": collection.count(), "sources": sorted(sources)}

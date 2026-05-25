"""Chroma vector store for document chunks."""

from __future__ import annotations

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

import config
from chunking import TextChunk


def get_chroma_client() -> chromadb.PersistentClient:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(
    client: chromadb.PersistentClient | None = None,
    collection_name: str | None = None,
) -> Collection:
    if client is None:
        client = get_chroma_client()
    name = collection_name or config.COLLECTION_NAME
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def clear_collection(collection_name: str | None = None) -> None:
    client = get_chroma_client()
    name = collection_name or config.COLLECTION_NAME
    try:
        client.delete_collection(name)
    except (ValueError, NotFoundError):
        pass


def add_chunks(
    collection: Collection,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")
    if not chunks:
        return

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_emb = embeddings[i : i + batch_size]
        collection.add(
            ids=[c.chunk_id for c in batch_chunks],
            documents=[c.content for c in batch_chunks],
            embeddings=batch_emb,
            metadatas=[
                {
                    "source_file": c.source_file,
                    "page_index": c.page_index,
                    "chunk_method": c.chunk_method,
                    "chunk_index": c.chunk_index,
                }
                for c in batch_chunks
            ],
        )


def collection_vector_count(collection_name: str) -> int:
    """Return stored vector count, or 0 if the collection is missing or empty."""
    try:
        return get_collection(collection_name=collection_name).count()
    except Exception:
        return 0


def query_collection(
    collection: Collection,
    query_embedding: list[float],
    top_k: int,
) -> dict:
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

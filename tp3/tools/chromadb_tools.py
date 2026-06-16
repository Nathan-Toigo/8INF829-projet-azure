"""ChromaDB access layer with OpenRouter embeddings.

Four on-disk collections (spec section 8) share one persistent client and one
embedding function pointed at OpenRouter's OpenAI-compatible embeddings endpoint:

- ``patient_documents``  - embedded chunks from uploaded records.
- ``clinical_guidelines`` - curated/seeded clinical guidance.
- ``similar_cases``      - historical/synthetic case patterns.
- ``agent_learnings``    - reusable successful patterns.

Search returns ranked hits with provenance (adapted from ``dms-2/rag``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.utils import embedding_functions

from config import settings

PATIENT_DOCUMENTS = "patient_documents"
CLINICAL_GUIDELINES = "clinical_guidelines"
SIMILAR_CASES = "similar_cases"
AGENT_LEARNINGS = "agent_learnings"

ALL_COLLECTIONS = [
    PATIENT_DOCUMENTS,
    CLINICAL_GUIDELINES,
    SIMILAR_CASES,
    AGENT_LEARNINGS,
]

_client: chromadb.api.ClientAPI | None = None
_ef = None


@dataclass
class SearchHit:
    rank: int
    content: str
    metadata: dict = field(default_factory=dict)
    distance: float = 0.0
    similarity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "content": self.content,
            "metadata": self.metadata,
            "distance": round(self.distance, 4),
            "similarity": round(self.similarity, 4),
        }


def _embedding_function():
    global _ef
    if _ef is None:
        settings.require_openrouter_key()
        # Chroma's OpenAI embedding function is OpenAI-compatible; pointing
        # api_base at OpenRouter lets a single key serve embeddings too.
        _ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.OPENROUTER_API_KEY,
            api_base=settings.OPENROUTER_BASE_URL,
            model_name=settings.OPENROUTER_EMBEDDING_MODEL,
        )
    return _ef


def _get_client() -> chromadb.api.ClientAPI:
    global _client
    if _client is None:
        settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    return _client


def get_collection(name: str):
    return _get_client().get_or_create_collection(
        name=name,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    collection_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict] | None = None,
) -> int:
    if not documents:
        return 0
    metadatas = metadatas or [{} for _ in documents]
    # Chroma rejects empty metadata dicts in some versions; ensure a key exists.
    metadatas = [m or {"_": "1"} for m in metadatas]
    get_collection(collection_name).upsert(
        ids=ids, documents=documents, metadatas=metadatas
    )
    return len(documents)


def add_chunks(
    collection_name: str,
    source: str,
    chunks: list[str],
    base_metadata: dict | None = None,
) -> int:
    if not chunks:
        return 0
    base = base_metadata or {}
    ids = [f"{source}::chunk::{i}" for i in range(len(chunks))]
    metadatas = [
        {**base, "source": source, "chunk_index": i} for i in range(len(chunks))
    ]
    return add_documents(collection_name, ids, chunks, metadatas)


def search(
    collection_name: str,
    query: str,
    k: int | None = None,
    where: dict | None = None,
) -> list[SearchHit]:
    collection = get_collection(collection_name)
    if collection.count() == 0:
        return []
    k = k or settings.RAG_TOP_K
    res = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        where=where or None,
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    hits: list[SearchHit] = []
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        hits.append(
            SearchHit(
                rank=rank,
                content=doc,
                metadata=dict(meta or {}),
                distance=float(dist),
                similarity=1.0 - float(dist),
            )
        )
    return hits


def stats(collection_name: str) -> dict:
    collection = get_collection(collection_name)
    return {"collection": collection_name, "count": collection.count()}


def all_stats() -> dict[str, int]:
    return {name: get_collection(name).count() for name in ALL_COLLECTIONS}

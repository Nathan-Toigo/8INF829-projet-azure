import qdrant_client
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    EMBED_MODEL_NAME,
)


def setup_embeddings():
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)


def get_qdrant_client():
    return qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def build_index(nodes: List[BaseNode], collection_name: str = QDRANT_COLLECTION) -> VectorStoreIndex:
    setup_embeddings()
    client = get_qdrant_client()

    # Drop and recreate the collection for a clean rebuild
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index


def load_index(collection_name: str = QDRANT_COLLECTION) -> VectorStoreIndex:
    """Reload an existing index from Qdrant without re-embedding."""
    setup_embeddings()
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    return VectorStoreIndex.from_vector_store(vector_store)


if __name__ == "__main__":
    from ingestion import load_all_documents
    from chunking import chunk_recursive

    docs = load_all_documents()
    nodes = chunk_recursive(docs)
    print(f"Indexing {len(nodes)} chunks into Qdrant collection '{QDRANT_COLLECTION}'...")
    index = build_index(nodes)
    print("Index built successfully.")
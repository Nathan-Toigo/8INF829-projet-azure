from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_recursive(documents: List[Document]) -> List[BaseNode]:
    """Recursive chunking (sentence-aware, fixed size with overlap)."""
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes


if __name__ == "__main__":
    from ingestion import load_all_documents
    docs = load_all_documents()
    nodes = chunk_recursive(docs)
    print(f"{len(docs)} documents -> {len(nodes)} chunks")
    print(f"Sample chunk metadata: {nodes[0].metadata}")
    print(f"Sample chunk text (first 200 chars): {nodes[0].text[:200]}")
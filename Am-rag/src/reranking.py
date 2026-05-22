from typing import List
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, QueryBundle


def get_reranker(top_n: int = 3, model: str = "BAAI/bge-reranker-base"):
    """Cross-encoder reranker. Use bge-reranker-base for speed; bge-reranker-v2-m3 for quality."""
    return SentenceTransformerRerank(model=model, top_n=top_n)


def rerank(reranker, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
    bundle = QueryBundle(query_str=query)
    return reranker.postprocess_nodes(nodes, query_bundle=bundle)


if __name__ == "__main__":
    from ingestion import load_all_documents
    from chunking import chunk_recursive
    from indexing import load_index
    from retrieval import get_hybrid_retriever

    docs = load_all_documents()
    nodes = chunk_recursive(docs)
    index = load_index()

    query = "Was the dominant lymph node considered malignant?"

    retriever = get_hybrid_retriever(index, nodes, top_k=10)
    retrieved = retriever.retrieve(query)
    print(f"Retrieved {len(retrieved)} nodes before reranking:")
    for n in retrieved:
        print(f"  [{n.score:.4f}] {n.metadata.get('source')} p.{n.metadata.get('page','-')}")

    reranker = get_reranker(top_n=5)
    reranked = rerank(reranker, retrieved, query)
    print(f"\nAfter reranking (top {len(reranked)}):")
    for n in reranked:
        print(f"  [{n.score:.4f}] {n.metadata.get('source')} p.{n.metadata.get('page','-')}")
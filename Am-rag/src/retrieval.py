from typing import List, Dict
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever

from config import TOP_K


# ---------- Dense ----------
def get_dense_retriever(index: VectorStoreIndex, top_k: int = TOP_K) -> VectorIndexRetriever:
    return VectorIndexRetriever(index=index, similarity_top_k=top_k)


# ---------- BM25 ----------
def get_bm25_retriever(nodes, top_k: int = TOP_K) -> BM25Retriever:
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)


# ---------- Hybrid via Reciprocal Rank Fusion ----------
class HybridRRFRetriever(BaseRetriever):
    """Hybrid retriever combining dense and BM25 via Reciprocal Rank Fusion."""

    def __init__(self, dense_retriever, bm25_retriever, top_k: int = TOP_K, k: int = 60):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        self.k = k  # RRF constant, 60 is the canonical default
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        dense_nodes = self.dense_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

        scores: Dict[str, float] = {}
        node_map: Dict[str, NodeWithScore] = {}

        for rank, n in enumerate(dense_nodes):
            nid = n.node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (self.k + rank + 1)
            node_map[nid] = n

        for rank, n in enumerate(bm25_nodes):
            nid = n.node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (self.k + rank + 1)
            if nid not in node_map:
                node_map[nid] = n

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        results = []
        for nid, fused_score in ranked:
            n = node_map[nid]
            results.append(NodeWithScore(node=n.node, score=fused_score))
        return results


def get_hybrid_retriever(index: VectorStoreIndex, nodes, top_k: int = TOP_K) -> HybridRRFRetriever:
    dense = get_dense_retriever(index, top_k=top_k * 2)
    bm25 = get_bm25_retriever(nodes, top_k=top_k * 2)
    return HybridRRFRetriever(dense, bm25, top_k=top_k)


if __name__ == "__main__":
    from ingestion import load_all_documents
    from chunking import chunk_recursive
    from indexing import load_index

    docs = load_all_documents()
    nodes = chunk_recursive(docs)
    index = load_index()

    query = "Was the dominant lymph node considered malignant?"

    print("=== Dense ===")
    for n in get_dense_retriever(index).retrieve(query):
        print(f"  [{n.score:.3f}] {n.metadata.get('source')} p.{n.metadata.get('page','-')}")

    print("\n=== BM25 ===")
    for n in get_bm25_retriever(nodes).retrieve(query):
        print(f"  [{n.score:.3f}] {n.metadata.get('source')} p.{n.metadata.get('page','-')}")

    print("\n=== Hybrid RRF ===")
    for n in get_hybrid_retriever(index, nodes).retrieve(query):
        print(f"  [{n.score:.4f}] {n.metadata.get('source')} p.{n.metadata.get('page','-')}")
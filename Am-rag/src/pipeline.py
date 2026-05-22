from dataclasses import dataclass
from typing import Optional

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.azure_openai import AzureOpenAI

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_CHAT_DEPLOYMENT,
    AZURE_MODEL_ALIAS,
    TOP_K,
)
from ingestion import load_all_documents
from chunking import chunk_recursive
from indexing import build_index, load_index
from retrieval import (
    get_dense_retriever,
    get_bm25_retriever,
    get_hybrid_retriever,
)
from reranking import get_reranker


@dataclass
class RAGConfig:
    retrieval_mode: str = "dense"   # "dense" | "bm25" | "hybrid"
    use_reranker: bool = False
    retrieval_top_k: int = TOP_K
    rerank_top_n: int = 3
    name: str = "baseline_dense"


def setup_llm():
    Settings.llm = AzureOpenAI(
        model=AZURE_MODEL_ALIAS,
        deployment_name=AZURE_CHAT_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def build_pipeline(cfg: RAGConfig, rebuild: bool = False):
    setup_llm()

    docs = load_all_documents()
    nodes = chunk_recursive(docs)

    if rebuild:
        index = build_index(nodes)
        print(f"[{cfg.name}] Rebuilt index: {len(nodes)} chunks from {len(docs)} docs.")
    else:
        index = load_index()

    # Retrieval
    # For BM25/hybrid we ask for more candidates if reranking, so the reranker has room to filter.
    base_top_k = cfg.retrieval_top_k * 3 if cfg.use_reranker else cfg.retrieval_top_k

    if cfg.retrieval_mode == "dense":
        retriever = get_dense_retriever(index, top_k=base_top_k)
    elif cfg.retrieval_mode == "bm25":
        retriever = get_bm25_retriever(nodes, top_k=base_top_k)
    elif cfg.retrieval_mode == "hybrid":
        retriever = get_hybrid_retriever(index, nodes, top_k=base_top_k)
    else:
        raise ValueError(f"Unknown retrieval_mode: {cfg.retrieval_mode}")

    # Reranking via node postprocessor
    node_postprocessors = []
    if cfg.use_reranker:
        node_postprocessors.append(get_reranker(top_n=cfg.rerank_top_n))

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=node_postprocessors,
    )
    return query_engine


# Predefined configurations to compare
CONFIGS = {
    "dense_top5": RAGConfig(retrieval_mode="dense", use_reranker=False, name="dense_top5"),
    "bm25_top5": RAGConfig(retrieval_mode="bm25", use_reranker=False, name="bm25_top5"),
    "hybrid_top5": RAGConfig(retrieval_mode="hybrid", use_reranker=False, name="hybrid_top5"),
    "hybrid_rerank_top3": RAGConfig(retrieval_mode="hybrid", use_reranker=True, rerank_top_n=3, name="hybrid_rerank_top3"),
    "dense_rerank_top3": RAGConfig(retrieval_mode="dense", use_reranker=True, rerank_top_n=3, name="dense_rerank_top3"),
}


def answer(query_engine, question: str):
    response = query_engine.query(question)
    print(f"\nQ: {question}")
    print(f"A: {response}\n")
    print("Sources:")
    for n in response.source_nodes:
        print(f"  - {n.metadata.get('source')} p.{n.metadata.get('page', '-')} (score={n.score:.4f})")
    return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG clinique - mode interactif")
    parser.add_argument(
        "--config",
        default="hybrid_rerank_top3",
        choices=list(CONFIGS.keys()),
        help="Configuration RAG à utiliser",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstruire l'index (à utiliser après ajout de nouveaux documents)",
    )
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    print(f"Configuration: {cfg.name}")
    qe = build_pipeline(cfg, rebuild=args.rebuild)
    print("\nPipeline prêt. Tape ta question (ou 'exit' / 'quit' pour quitter).\n")

    while True:
        try:
            question = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if question.lower() in ("exit", "quit", ""):
            print("Au revoir.")
            break

        answer(qe, question)
        print("-" * 80)
import argparse
from pathlib import Path

import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.azure_openai import AzureOpenAI

from config import (
    QDRANT_HOST, QDRANT_PORT, EMBED_MODEL_NAME, TOP_K,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION, AZURE_CHAT_DEPLOYMENT, AZURE_MODEL_ALIAS,
)
from ingestion import load_all_documents
from chunking import chunk_documents, CHUNKERS

DEFAULT_QUESTIONS_FILE = Path(__file__).parent.parent / "evaluation" / "team_questions.txt"


def setup_models():
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = AzureOpenAI(
        model=AZURE_MODEL_ALIAS,
        deployment_name=AZURE_CHAT_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def build_engine_for_strategy(strategy: str, docs):
    """Construit un index dedie pour une strategie de chunking et renvoie un query engine (dense simple)."""
    nodes = chunk_documents(docs, strategy)
    client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collection = f"chunk_{strategy}"
    if client.collection_exists(collection):
        client.delete_collection(collection)
    vector_store = QdrantVectorStore(client=client, collection_name=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)
    engine = RetrieverQueryEngine.from_args(retriever=retriever)
    return engine, len(nodes)


def load_questions(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Comparatif des strategies de chunking")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_FILE))
    parser.add_argument("--strategies", nargs="+", default=list(CHUNKERS.keys()))
    parser.add_argument("--out", default=str(Path(__file__).parent.parent / "evaluation" / "comparatif_chunking.md"))
    args = parser.parse_args()

    setup_models()
    docs = load_all_documents()
    questions = load_questions(Path(args.questions))
    print(f"{len(questions)} questions, {len(args.strategies)} strategies de chunking.")

    engines = {}
    for strat in args.strategies:
        print(f"Indexation pour la strategie: {strat}")
        engine, n_chunks = build_engine_for_strategy(strat, docs)
        engines[strat] = engine
        print(f"  -> {n_chunks} chunks")

    lines = ["# Comparatif des strategies de chunking (retrieval dense fixe)\n"]
    for qi, question in enumerate(questions, start=1):
        print(f"\n=== Q{qi}: {question} ===")
        lines.append(f"\n## Q{qi}. {question}\n")
        for strat in args.strategies:
            response = engines[strat].query(question)
            answer_text = str(response).strip()
            srcs = ", ".join(
                f"{n.metadata.get('source')} p.{n.metadata.get('page', '-')}"
                for n in response.source_nodes
            )
            print(f"  [{strat}] {answer_text[:80]}...")
            lines.append(f"**{strat}**")
            lines.append(f"{answer_text}")
            lines.append(f"_Sources : {srcs}_\n")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nComparatif ecrit dans : {out_path}")


if __name__ == "__main__":
    main()
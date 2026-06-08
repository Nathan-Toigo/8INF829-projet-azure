from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL_NAME


def chunk_by_tokens(documents: List[Document]) -> List[BaseNode]:
    """Decoupage par taille de tokens (512 tokens, overlap 64)."""
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.get_nodes_from_documents(documents)


def chunk_by_words(documents: List[Document], n_words: int = 250) -> List[BaseNode]:
    """Decoupage par nombre de mots (~250 mots).

    SentenceSplitter raisonne en tokens. On approxime: 1 mot ~ 1.3 tokens,
    donc 250 mots ~ 325 tokens. On garde un petit overlap proportionnel.
    """
    approx_tokens = int(n_words * 1.3)
    splitter = SentenceSplitter(
        chunk_size=approx_tokens,
        chunk_overlap=int(approx_tokens * 0.12),
    )
    return splitter.get_nodes_from_documents(documents)


def chunk_semantic(documents: List[Document]) -> List[BaseNode]:
    """Decoupage semantique: coupe quand le sens change (similarite des embeddings)."""
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model,
    )
    return splitter.get_nodes_from_documents(documents)


# Registre des strategies, accessible par nom
CHUNKERS = {
    "words250": chunk_by_words,
    "tokens512": chunk_by_tokens,
    "semantic": chunk_semantic,
}


def chunk_documents(documents: List[Document], strategy: str) -> List[BaseNode]:
    if strategy not in CHUNKERS:
        raise ValueError(f"Strategie inconnue: {strategy}. Choix: {list(CHUNKERS)}")
    return CHUNKERS[strategy](documents)


if __name__ == "__main__":
    from ingestion import load_all_documents

    docs = load_all_documents()
    for name in CHUNKERS:
        nodes = chunk_documents(docs, name)
        avg_len = sum(len(n.text) for n in nodes) / len(nodes)
        print(f"{name:12s} -> {len(nodes):3d} chunks, longueur moyenne {avg_len:.0f} caracteres")
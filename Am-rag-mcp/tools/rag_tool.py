"""
Outil MCP: RAG clinique.
Expose une fonction search_clinical_documents qui interroge le RAG Phase 1
(Qdrant + hybride + reranker + Azure gpt-4.1-mini).
"""
import sys
from pathlib import Path

# Ajoute le src de la Phase 1 pour reutiliser le pipeline existant
PHASE1_SRC = Path(__file__).parent.parent.parent / "Am-rag" / "src"
sys.path.insert(0, str(PHASE1_SRC))

from mcp.server.fastmcp import FastMCP
from pipeline import build_pipeline, CONFIGS

# Initialisation unique du pipeline (cout au premier appel seulement)
print("Initialisation du RAG clinique...", file=sys.stderr)
_query_engine = build_pipeline(CONFIGS["hybrid_rerank_top3"], rebuild=False)
print("RAG pret.", file=sys.stderr)

mcp = FastMCP("RAG Clinique")


@mcp.tool()
def search_clinical_documents(question: str) -> dict:
    """
    Recherche dans les documents cliniques du patient et genere une reponse
    fondee sur les sources retrouvees.

    Args:
        question: Question clinique en langage naturel (FR ou EN).

    Returns:
        Dict avec 'answer' (str), 'sources' (list de dicts source/page/score).
    """
    response = _query_engine.query(question)
    sources = [
        {
            "source": n.metadata.get("source"),
            "page": n.metadata.get("page"),
            "score": float(n.score) if n.score is not None else None,
        }
        for n in response.source_nodes
    ]
    return {
        "answer": str(response),
        "sources": sources,
    }


if __name__ == "__main__":
    mcp.run()
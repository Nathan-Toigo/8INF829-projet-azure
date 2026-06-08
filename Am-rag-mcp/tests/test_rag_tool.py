"""Test direct de la fonction RAG sans passer par MCP (validation du wiring)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

# On importe la fonction directement pour la tester
from rag_tool import search_clinical_documents

if __name__ == "__main__":
    questions = [
        "What medications is the patient taking?",
        "What were the findings of the neck ultrasound?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        result = search_clinical_documents(q)
        print(f"A: {result['answer']}")
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['source']} p.{s['page']} (score={s['score']:.4f})")
        print("-" * 80)
        
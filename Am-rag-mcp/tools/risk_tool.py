"""
Outil MCP: Detection de risques cliniques (red flags).
Analyse le dossier patient au demarrage et identifie les elements
necessitant attention, classes par niveau d'urgence.
"""
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Reutilise l'ingestion de la Phase 1
PHASE1_SRC = Path(__file__).parent.parent.parent / "Am-rag" / "src"
sys.path.insert(0, str(PHASE1_SRC))

from ingestion import load_all_documents
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_CHAT_DEPLOYMENT,
)
from openai import AzureOpenAI as AzureClient
from mcp.server.fastmcp import FastMCP


RISK_PROMPT = """You are a clinical assistant analyzing a patient record to identify clinical risks and red flags before a physician consultation.

Analyze the documents below and identify clinical risks, abnormalities, missing follow-ups, concerning trends, and inconsistencies. Be thorough but precise: focus on findings supported by the documents, not generic concerns.

Return a JSON object with the following structure:

{{
  "high_priority": [
    {{
      "issue": "short title of the risk",
      "details": "1-2 sentence explanation with specific values or dates from the documents",
      "evidence": "brief quote or reference to source document",
      "recommended_action": "what should be done"
    }}
  ],
  "medium_priority": [
    {{ same structure }}
  ],
  "low_priority": [
    {{ same structure }}
  ],
  "missing_or_overdue": [
    {{
      "item": "what is missing or overdue",
      "expected": "when it was expected",
      "rationale": "why it matters"
    }}
  ],
  "abnormal_lab_values": [
    {{
      "test": "test name",
      "value": "specific value with units",
      "date": "date",
      "reference_or_concern": "why this is abnormal"
    }}
  ],
  "overall_risk_assessment": "2-3 sentence overall clinical risk summary"
}}

Priority levels:
- high_priority: urgent or potentially serious findings requiring prompt attention
- medium_priority: notable findings requiring follow-up within weeks
- low_priority: minor abnormalities or items to monitor

Base every risk on actual documented findings. Cite specific values, dates, and sources when possible.
Return ONLY valid JSON, no preamble, no markdown fences.

PATIENT DOCUMENTS:
{documents}
"""


def _build_documents_text(docs, max_chars=80000):
    parts = []
    total = 0
    for d in docs:
        header = f"\n\n=== Source: {d.metadata.get('source')} (page {d.metadata.get('page', '-')}) ===\n"
        chunk = header + d.text
        if total + len(chunk) > max_chars:
            parts.append(chunk[: max_chars - total])
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts)


def _generate_risk_analysis() -> dict:
    print("Chargement des documents patient...", file=sys.stderr)
    docs = load_all_documents()
    print(f"  -> {len(docs)} documents charges.", file=sys.stderr)

    documents_text = _build_documents_text(docs)
    prompt = RISK_PROMPT.format(documents=documents_text)

    print("Analyse des risques via Azure...", file=sys.stderr)
    client = AzureClient(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    response = client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip("` \n")

    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Avertissement: JSON parse a echoue. {e}", file=sys.stderr)
        analysis = {"raw_text": raw}

    print("Analyse des risques prete.", file=sys.stderr)
    return analysis


# Generation unique au demarrage
# _RISKS = _generate_risk_analysis()


# # ---------- MCP server ----------
# mcp = FastMCP("Detection de risques")


# @mcp.tool()
# def get_clinical_risks(priority: str = "all") -> dict:
#     """
#     Retourne l'analyse des risques cliniques identifies dans le dossier patient.

#     Args:
#         priority: Niveau de risque a retourner. Valeurs possibles:
#                   'all' (defaut), 'high', 'medium', 'low',
#                   'missing_or_overdue', 'abnormal_labs', 'overall'.

#     Returns:
#         Dict contenant les risques filtres.
#     """
#     if priority == "all":
#         return _RISKS

#     mapping = {
#         "high": "high_priority",
#         "medium": "medium_priority",
#         "low": "low_priority",
#         "missing_or_overdue": "missing_or_overdue",
#         "abnormal_labs": "abnormal_lab_values",
#         "overall": "overall_risk_assessment",
#     }

#     if priority in mapping:
#         key = mapping[priority]
#         return {key: _RISKS.get(key, [])}

#     return {
#         "error": f"Priority '{priority}' inconnue.",
#         "available": list(mapping.keys()) + ["all"],
#     }


# if __name__ == "__main__":
#     mcp.run()

_RISKS = None


def _ensure_risks():
    global _RISKS
    if _RISKS is None:
        _RISKS = _generate_risk_analysis()
    return _RISKS


# ---------- MCP server ----------
mcp = FastMCP("Detection de risques")


@mcp.tool()
def get_clinical_risks(priority: str = "all") -> dict:
    """
    Retourne l'analyse des risques cliniques identifies dans le dossier patient.

    Args:
        priority: Niveau de risque a retourner. Valeurs possibles:
                  'all' (defaut), 'high', 'medium', 'low',
                  'missing_or_overdue', 'abnormal_labs', 'overall'.

    Returns:
        Dict contenant les risques filtres.
    """
    risks = _ensure_risks()
    if priority == "all":
        return risks

    mapping = {
        "high": "high_priority",
        "medium": "medium_priority",
        "low": "low_priority",
        "missing_or_overdue": "missing_or_overdue",
        "abnormal_labs": "abnormal_lab_values",
        "overall": "overall_risk_assessment",
    }

    if priority in mapping:
        key = mapping[priority]
        return {key: risks.get(key, [])}

    return {
        "error": f"Priority '{priority}' inconnue.",
        "available": list(mapping.keys()) + ["all"],
    }


if __name__ == "__main__":
    mcp.run()
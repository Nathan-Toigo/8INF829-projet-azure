"""
Outil MCP: Resume patient.
Genere une synthese structuree du dossier patient au demarrage,
puis la retourne sur demande via MCP.
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


SUMMARY_PROMPT = """You are a clinical assistant preparing a structured patient summary for a physician before a consultation.

Based on the patient documents provided below, produce a concise structured summary in JSON format with the following fields:

- patient_identity: age, sex, MRN if available
- main_concern: primary reason for ongoing follow-up
- chronic_conditions: list of known chronic conditions
- past_medical_history: relevant past events (surgeries, hospitalizations, major diagnoses)
- current_medications: list of regular medications (or "none documented" if absent)
- allergies: known allergies or "none documented"
- recent_investigations: list of recent exams with brief findings (last 12-24 months)
- ongoing_followup: pending tests, scheduled visits, surveillance plans
- key_lab_trends: notable lab evolutions
- clinical_summary_paragraph: 3-4 sentences synthesizing the clinical picture for the physician

Be precise, use medical terminology, cite specific dates and values when available.
Return ONLY valid JSON, no preamble, no markdown fences.

PATIENT DOCUMENTS:
{documents}
"""


def _build_documents_text(docs, max_chars=80000):
    """Concatene les documents avec un marqueur par source. Tronque si necessaire."""
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


def _generate_summary() -> dict:
    """Appel Azure pour produire le resume une seule fois au demarrage."""
    print("Chargement des documents patient...", file=sys.stderr)
    docs = load_all_documents()
    print(f"  -> {len(docs)} documents charges.", file=sys.stderr)

    documents_text = _build_documents_text(docs)
    prompt = SUMMARY_PROMPT.format(documents=documents_text)

    print("Generation du resume patient via Azure...", file=sys.stderr)
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

    # Nettoyer les eventuelles fences markdown au cas ou
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip("` \n")

    try:
        summary = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Avertissement: JSON parse a echoue, retour texte brut. {e}", file=sys.stderr)
        summary = {"raw_text": raw}

    print("Resume patient pret.", file=sys.stderr)
    return summary


# Generation unique au demarrage
# _SUMMARY = _generate_summary()


# # ---------- MCP server ----------
# mcp = FastMCP("Resume Patient")


# @mcp.tool()
# def get_patient_summary(section: str = "all") -> dict:
#     """
#     Retourne le resume structure du dossier patient.

#     Args:
#         section: Section specifique a retourner ('all' par defaut).
#                  Valeurs possibles: all, patient_identity, main_concern, chronic_conditions,
#                  past_medical_history, current_medications, allergies, recent_investigations,
#                  ongoing_followup, key_lab_trends, clinical_summary_paragraph.

#     Returns:
#         Dict contenant le resume complet ou la section demandee.
#     """
#     if section == "all":
#         return _SUMMARY
#     if section in _SUMMARY:
#         return {section: _SUMMARY[section]}
#     return {
#         "error": f"Section '{section}' inconnue.",
#         "available_sections": list(_SUMMARY.keys()),
#     }


# if __name__ == "__main__":
#     mcp.run()

# Generation paresseuse: au premier appel de l'outil seulement
_SUMMARY = None


def _ensure_summary():
    global _SUMMARY
    if _SUMMARY is None:
        _SUMMARY = _generate_summary()
    return _SUMMARY


# ---------- MCP server ----------
mcp = FastMCP("Resume Patient")


@mcp.tool()
def get_patient_summary(section: str = "all") -> dict:
    """
    Retourne le resume structure du dossier patient.

    Args:
        section: Section specifique a retourner ('all' par defaut).
                 Valeurs possibles: all, patient_identity, main_concern, chronic_conditions,
                 past_medical_history, current_medications, allergies, recent_investigations,
                 ongoing_followup, key_lab_trends, clinical_summary_paragraph.

    Returns:
        Dict contenant le resume complet ou la section demandee.
    """
    summary = _ensure_summary()
    if section == "all":
        return summary
    if section in summary:
        return {section: summary[section]}
    return {
        "error": f"Section '{section}' inconnue.",
        "available_sections": list(summary.keys()),
    }


if __name__ == "__main__":
    mcp.run()
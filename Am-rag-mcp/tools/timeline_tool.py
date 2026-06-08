"""
Outil MCP: Timeline medicale.
Extrait les dates des documents cliniques et construit une chronologie structuree.
"""
import sys
import re
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

warnings.filterwarnings("ignore")

# Reutilise l'ingestion de la Phase 1
PHASE1_SRC = Path(__file__).parent.parent.parent / "Am-rag" / "src"
sys.path.insert(0, str(PHASE1_SRC))

from ingestion import load_all_documents
from dateutil import parser as date_parser
from mcp.server.fastmcp import FastMCP


# ---------- Regex pour reperer les dates ----------
# Formats supportes: 12/2021, 2021-10-12, 10/25/2021, October 2021, 2021
DATE_PATTERNS = [
    r"\b(\d{4}-\d{2}-\d{2})\b",                         # 2021-10-12
    r"\b(\d{1,2}/\d{1,2}/\d{4})\b",                     # 10/25/2021
    r"\b(\d{1,2}/\d{4})\b",                             # 12/2021
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
    r"\b(\d{4})\b",                                     # 2021 (annee seule, en dernier recours)
]
DATE_REGEX = re.compile("|".join(DATE_PATTERNS))


# ---------- Classification par type d'evenement ----------
EVENT_TYPES = {
    "imaging": ["ultrasound", "us neck", "mri", "ct", "scan", "echographie", "irm", "scanner"],
    "pathology": ["fna", "pathology", "biopsy", "cytology", "biopsie"],
    "consultation": ["consult", "consultation", "evaluation", "visite"],
    "lab": ["lab", "blood", "crp", "cbc", "vitamin", "serology", "laboratoire"],
    "endoscopy": ["endoscopy", "egd", "colonoscopy", "endoscopie"],
    "follow_up": ["follow-up", "followup", "suivi", "f/u"],
}


def classify_event(text: str, source_filename: str) -> str:
    """Classe un evenement selon les mots-cles trouves dans le contexte ou le nom de fichier."""
    combined = (text + " " + source_filename).lower()
    for event_type, keywords in EVENT_TYPES.items():
        if any(kw in combined for kw in keywords):
            return event_type
    return "other"


def normalize_date(raw: str) -> Optional[str]:
    """Convertit une date brute en ISO YYYY-MM-DD (ou YYYY-MM si jour absent)."""
    try:
        # 12/2021 -> on force le 1er du mois
        if re.match(r"^\d{1,2}/\d{4}$", raw):
            month, year = raw.split("/")
            return f"{year}-{int(month):02d}-01"
        # 2021 seul -> 2021-01-01, mais on garde une trace que c'est imprecis
        if re.match(r"^\d{4}$", raw):
            year = int(raw)
            if 2000 <= year <= 2030:
                return f"{year}-01-01"
            return None
        dt = date_parser.parse(raw, fuzzy=True, default=datetime(2000, 1, 1))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None


def extract_events_from_document(doc) -> List[Dict]:
    """Extrait tous les evenements dates d'un document, avec leur contexte."""
    text = doc.text
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page")

    events = []
    seen_dates = set()
    for match in DATE_REGEX.finditer(text):
        raw_date = match.group(0)
        iso_date = normalize_date(raw_date)
        if not iso_date:
            continue

        # Contexte: la phrase autour de la date
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 200)
        context = text[start:end].replace("\n", " ").strip()

        # Deduplication: meme date + meme contexte court = on saute
        key = (iso_date, context[:80])
        if key in seen_dates:
            continue
        seen_dates.add(key)

        events.append({
            "date": iso_date,
            "raw_date": raw_date,
            "event_type": classify_event(context, source),
            "source": source,
            "page": page,
            "context": context,
        })
    return events


# ---------- Construction de la timeline ----------
print("Construction de la timeline patient...", file=sys.stderr)
_all_documents = load_all_documents()

# Date de naissance a exclure (apparait dans les en-tetes de tous les docs)
PATIENT_DOB = "1981-08-19"

_timeline = []
for doc in _all_documents:
    _timeline.extend(extract_events_from_document(doc))

# Filtrer le DOB
_timeline = [e for e in _timeline if e["date"] != PATIENT_DOB]

# Deduplication globale : meme date + meme source = on garde un seul evenement
_seen = set()
_dedup = []
for e in _timeline:
    key = (e["date"], e["source"])
    if key not in _seen:
        _seen.add(key)
        _dedup.append(e)
_timeline = sorted(_dedup, key=lambda e: e["date"])

print(f"Timeline: {len(_timeline)} evenements dates extraits (apres deduplication).",
      file=sys.stderr)


# ---------- MCP server ----------
mcp = FastMCP("Timeline Patient")


@mcp.tool()
def get_patient_timeline(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    event_type: Optional[str] = None,
    max_results: int = 50,
) -> dict:
    """
    Retourne la chronologie des evenements medicaux du patient.

    Args:
        start_date: Date de debut au format YYYY-MM-DD (inclus). None = depuis le debut.
        end_date: Date de fin au format YYYY-MM-DD (inclus). None = jusqu'a aujourd'hui.
        event_type: Filtre par type. Valeurs: imaging, pathology, consultation, lab, endoscopy, follow_up, other.
        max_results: Nombre maximum d'evenements retournes (default 50).

    Returns:
        Dict avec 'count' (int) et 'events' (list of event dicts).
    """
    filtered = _timeline

    if start_date:
        filtered = [e for e in filtered if e["date"] >= start_date]
    if end_date:
        filtered = [e for e in filtered if e["date"] <= end_date]
    if event_type:
        filtered = [e for e in filtered if e["event_type"] == event_type]

    filtered = filtered[:max_results]
    return {
        "count": len(filtered),
        "events": filtered,
    }


if __name__ == "__main__":
    mcp.run()
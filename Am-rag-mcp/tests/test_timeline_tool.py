"""Test direct de la fonction Timeline sans passer par MCP."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from timeline_tool import get_patient_timeline

if __name__ == "__main__":
    print("\n=== Toute la timeline (10 premiers evenements) ===")
    result = get_patient_timeline(max_results=10)
    print(f"Total: {result['count']} evenements")
    for e in result["events"]:
        print(f"  [{e['date']}] ({e['event_type']:12s}) {e['source']}")
        print(f"      -> {e['context'][:120]}...")

    print("\n=== Filtre: imagerie en 2021 ===")
    result = get_patient_timeline(start_date="2021-01-01", end_date="2021-12-31",
                                   event_type="imaging")
    print(f"Total: {result['count']} evenements")
    for e in result["events"]:
        print(f"  [{e['date']}] {e['source']}")

    print("\n=== Filtre: tous les labs ===")
    result = get_patient_timeline(event_type="lab")
    print(f"Total: {result['count']} evenements")
    for e in result["events"][:5]:
        print(f"  [{e['date']}] {e['source']} p.{e['page']}")
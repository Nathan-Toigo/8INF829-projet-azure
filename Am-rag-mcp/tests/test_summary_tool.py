"""Test direct de l'outil de resume patient."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from summary_tool import get_patient_summary

if __name__ == "__main__":
    print("=== Resume complet ===")
    full = get_patient_summary()
    print(json.dumps(full, indent=2, ensure_ascii=False))

    print("\n=== Section: current_medications ===")
    print(json.dumps(get_patient_summary("current_medications"), indent=2, ensure_ascii=False))

    print("\n=== Section: main_concern ===")
    print(json.dumps(get_patient_summary("main_concern"), indent=2, ensure_ascii=False))
"""Test direct de l'outil de detection de risques."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from risk_tool import get_clinical_risks

if __name__ == "__main__":
    print("=== Analyse complete des risques ===")
    full = get_clinical_risks()
    print(json.dumps(full, indent=2, ensure_ascii=False))

    print("\n=== Risques de haute priorite ===")
    print(json.dumps(get_clinical_risks("high"), indent=2, ensure_ascii=False))

    print("\n=== Valeurs de laboratoire anormales ===")
    print(json.dumps(get_clinical_risks("abnormal_labs"), indent=2, ensure_ascii=False))

    print("\n=== Evaluation globale ===")
    print(json.dumps(get_clinical_risks("overall"), indent=2, ensure_ascii=False))
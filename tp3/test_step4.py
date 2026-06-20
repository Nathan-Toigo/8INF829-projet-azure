"""Test isolated des 3 agents Step 4 avec un state simule.

Lance: python test_step4.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

from agents.step_4_patient_explanation import run as patient_explanation
from agents.step_4_patient_representative import run as patient_representative
from agents.step_4_clinical_review import run as clinical_review


# State simule: ce qu'on aurait apres que Step 2 ait tourne
SIMULATED_STATE = {
    "patient_id": "test-patient-001",
    "patient_question": "Quels sont les points importants de mon dossier?",
    "intent": "patient_summary_request",
    "step_count": 4,
    "agents_run": [
        "1.1 Clinical Agent Orchestrator",
        "2.1 Timeline Agent",
        "2.2 Guidelines Agent",
        "2.3 Risk Agent",
        "2.4 Case Investigator Agent",
    ],
    # Sorties de Step 2 (simulees)
    "timeline": [
        {
            "date": "2021-10-18",
            "description": "Ultrasound neck: dominant left level II lymph node 32x10x28 mm",
            "confidence": 0.9,
            "source": "02_US_Neck_2021.pdf",
        },
        {
            "date": "2021-10-25",
            "description": "FNA cytopathology: reactive lymphoid population, negative for malignancy",
            "confidence": 0.95,
            "source": "03_FNA_Pathology_2021.pdf",
        },
        {
            "date": "2025-03-24",
            "description": "Follow-up ultrasound: dominant right level 2A node enlarged to 37.8x11.5x27.5 mm; CRP 9.1 mg/L",
            "confidence": 0.9,
            "source": "10_IM_Followup_2025.pdf",
        },
    ],
    "guidelines": [
        {
            "topic": "Persistent cervical lymphadenopathy",
            "recommendation": "Consider repeat biopsy when interval growth is observed after a negative FNA, due to possible sampling error.",
            "source": "ENT clinical practice guidelines",
            "relevance": 0.9,
        },
        {
            "topic": "Small pulmonary nodules <6 mm",
            "recommendation": "Routine follow-up not mandatory for low-risk patients; surveillance optional.",
            "source": "Fleischner Society 2017",
            "relevance": 0.7,
        },
    ],
    "guideline_sources": ["ENT guidelines", "Fleischner Society 2017"],
    "risks": [
        {
            "issue": "Enlarging persistent cervical lymphadenopathy",
            "severity": "high",
            "rationale": "Interval growth (32 -> 37.8 mm) over 4 years despite negative FNA in 2021. Sampling error possible.",
            "monitoring": "Repeat excisional biopsy recommended.",
        },
        {
            "issue": "Rising CRP",
            "severity": "moderate",
            "rationale": "CRP rose from 5.0 mg/L (Jan 2025) to 9.1 mg/L (Mar 2025), indicating low-grade inflammation.",
            "monitoring": "Repeat CRP and CBC; consider infectious workup.",
        },
    ],
    "red_flags": [
        "Enlarging cervical lymph node despite prior negative FNA",
        "Rising CRP without clear etiology",
    ],
    "risk_rationale": [
        "Persistence + growth + inflammatory markers = possible occult lymphoma or chronic infection",
    ],
    "similar_cases": [
        {
            "summary": "43yo male with persistent cervical lymphadenopathy, eventually diagnosed with indolent B-cell lymphoma after second biopsy.",
            "similarity": 0.75,
            "relevant_outcome": "Excisional biopsy yielded diagnosis missed by initial FNA.",
        },
    ],
    "case_patterns": [
        "Initial FNA negative + interval growth -> excisional biopsy often diagnostic",
    ],
}


def run_agent(name, agent, state):
    print(f"\n{'='*80}")
    print(f"=== Running {name}")
    print(f"{'='*80}")
    update = agent(state)
    print(f"\nStatus: {update.get('agent_trace', [{}])[-1].get('status')}")
    print(f"Next agent: {update.get('next_agent')}")
    print(f"Needs orchestrator: {update.get('needs_orchestrator')}")
    print(f"\nMemory updates (keys): {sorted(update.get('agent_trace', [{}])[-1].get('memory_update_keys', []))}")
    # Affiche les nouvelles cles ajoutees a l'etat
    new_keys = set(update.keys()) - set(state.keys())
    interesting = [k for k in new_keys if not k.startswith("_")
                   and k not in {"agent_trace", "tool_calls", "token_ledger",
                                  "errors", "step_count", "agents_run",
                                  "next_agent", "needs_orchestrator",
                                  "long_term_context"}]
    for k in sorted(interesting):
        v = update[k]
        s = json.dumps(v, ensure_ascii=False, indent=2, default=str)
        print(f"\n--- {k} ---")
        print(s[:2000] + ("..." if len(s) > 2000 else ""))
    return {**state, **update}


def main():
    state = SIMULATED_STATE

    # Etape 4.1
    state = run_agent("4.1 Patient Explanation", patient_explanation, state)
    if not state.get("patient_explanation"):
        print("\n[ABORT] 4.1 n'a pas produit de patient_explanation, on arrete.")
        return

    # Etape 4.2
    state = run_agent("4.2 Patient Representative", patient_representative, state)

    # Etape 4.3
    state = run_agent("4.3 Clinical Review", clinical_review, state)

    # Resume final
    print(f"\n{'='*80}")
    print("=== RESUME FINAL")
    print(f"{'='*80}")
    print(f"\n>>> Patient explanation final:\n{state.get('patient_explanation', '')}")
    print(f"\n>>> Appropriateness passed: {state.get('patient_appropriateness_passed')}")
    print(f"\n>>> Clinical review passed: {state.get('clinical_review_passed')}")
    print(f"\n>>> Total steps: {state.get('step_count')}")
    print(f">>> Agents run: {state.get('agents_run')}")


if __name__ == "__main__":
    main()
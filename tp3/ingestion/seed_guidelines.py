"""Seed a handful of synthetic clinical guideline snippets into ChromaDB.

Gives the ``2.2 Guidelines Agent`` retrievable context out of the box. Run via
``python -m ingestion.seed_guidelines`` or the helper button on the Upload page.

Synthetic, simplified guidance for software validation only - NOT clinical advice.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import chromadb_tools

GUIDELINES: list[dict] = [
    {
        "id": "guideline-dm2-monitoring",
        "topic": "Type 2 diabetes monitoring",
        "source": "Synthetic ADA-style summary",
        "text": (
            "For adults with type 2 diabetes, check HbA1c at least twice yearly "
            "when at goal and quarterly when therapy changes or glycemic control "
            "is inadequate. A general HbA1c target is <7.0% for many non-pregnant "
            "adults; individualize for frailty and hypoglycemia risk. Screen "
            "annually for nephropathy (urine albumin-to-creatinine ratio, eGFR), "
            "retinopathy, and perform foot exams."
        ),
    },
    {
        "id": "guideline-htn-management",
        "topic": "Hypertension management",
        "source": "Synthetic ACC/AHA-style summary",
        "text": (
            "Diagnose hypertension with confirmed BP >=130/80 mmHg. First-line "
            "agents include ACE inhibitors or ARBs, thiazide-type diuretics, and "
            "calcium channel blockers. Reassess BP and adherence within 1 month "
            "of initiating or changing therapy. Monitor electrolytes and renal "
            "function after starting ACE inhibitors/ARBs or diuretics."
        ),
    },
    {
        "id": "guideline-thyroid-nodule",
        "topic": "Thyroid nodule workup",
        "source": "Synthetic ATA-style summary",
        "text": (
            "Evaluate thyroid nodules with TSH and neck ultrasound. Nodules with "
            "suspicious sonographic features or size thresholds warrant fine-needle "
            "aspiration (FNA) biopsy. Indeterminate cytology may require repeat FNA, "
            "molecular testing, or diagnostic lobectomy. Follow benign nodules with "
            "periodic ultrasound."
        ),
    },
    {
        "id": "guideline-ckd-monitoring",
        "topic": "Chronic kidney disease monitoring",
        "source": "Synthetic KDIGO-style summary",
        "text": (
            "Stage CKD by eGFR and albuminuria. Monitor eGFR and urine "
            "albumin-to-creatinine ratio at a frequency guided by risk category. "
            "Use ACE inhibitors or ARBs for albuminuric CKD, avoid nephrotoxins, "
            "and adjust renally-cleared medication doses. Review potassium and "
            "bicarbonate periodically."
        ),
    },
    {
        "id": "guideline-anticoagulation-afib",
        "topic": "Anticoagulation in atrial fibrillation",
        "source": "Synthetic guideline summary",
        "text": (
            "Estimate stroke risk in atrial fibrillation with CHA2DS2-VASc and "
            "bleeding risk with HAS-BLED. Offer oral anticoagulation when stroke "
            "risk is elevated, preferring direct oral anticoagulants over warfarin "
            "for most patients. Reassess renal function regularly as it affects DOAC "
            "dosing."
        ),
    },
    {
        "id": "guideline-lung-nodule-followup",
        "topic": "Incidental pulmonary nodule follow-up",
        "source": "Synthetic Fleischner-style summary",
        "text": (
            "Manage incidentally detected pulmonary nodules by size, attenuation, "
            "and patient risk. Small solid nodules in low-risk patients may need no "
            "routine follow-up, while larger or higher-risk nodules require interval "
            "CT surveillance or further evaluation. Document comparison with prior "
            "imaging when available."
        ),
    },
]


def seed() -> int:
    ids = [g["id"] for g in GUIDELINES]
    docs = [g["text"] for g in GUIDELINES]
    metas = [
        {"topic": g["topic"], "source": g["source"]} for g in GUIDELINES
    ]
    return chromadb_tools.add_documents(
        chromadb_tools.CLINICAL_GUIDELINES, ids, docs, metas
    )


if __name__ == "__main__":
    count = seed()
    print(f"Seeded {count} clinical guideline snippets into ChromaDB.")

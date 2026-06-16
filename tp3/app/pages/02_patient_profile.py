"""Page 2 - Patient Profile."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.shared import page_setup, require_patient, sidebar
from tools import mongodb_tools

page_setup("Patient Profile")
sidebar()

st.title("Patient profile")

pid = require_patient()
if pid:
    patient = mongodb_tools.get_patient(pid) or {"_id": pid}
    documents = mongodb_tools.list_documents(pid)
    resources = mongodb_tools.list_clinical_resources(pid)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(documents))
    c2.metric("Clinical resources", len(resources))
    c3.metric("Conditions", sum(1 for r in resources if r.get("resourceType") == "Condition"))
    c4.metric(
        "Medications",
        sum(1 for r in resources if r.get("resourceType") == "MedicationStatement"),
    )

    st.subheader("Demographics")
    st.write(
        {
            "id": patient.get("_id"),
            "name": patient.get("name", "Redacted"),
            "birthDate": patient.get("birthDate", ""),
            "sex": patient.get("sex", ""),
        }
    )

    def _by_type(rtype: str):
        rows = [
            {
                "name": (r.get("code") or {}).get("display", ""),
                "value": r.get("value", ""),
                "unit": r.get("unit", ""),
                "date": r.get("effectiveDate", ""),
                "source": r.get("sourceDocumentId", ""),
            }
            for r in resources
            if r.get("resourceType") == rtype
        ]
        return rows

    st.subheader("Known conditions")
    st.dataframe(_by_type("Condition"), use_container_width=True)

    st.subheader("Medications")
    st.dataframe(_by_type("MedicationStatement"), use_container_width=True)

    st.subheader("Allergies")
    st.dataframe(_by_type("AllergyIntolerance"), use_container_width=True)

    st.subheader("Recent labs / observations")
    st.dataframe(_by_type("Observation"), use_container_width=True)

    st.subheader("Source documents")
    st.dataframe(
        [
            {
                "file": d.get("fileName"),
                "type": d.get("documentType"),
                "ocr": d.get("ocrMethod"),
                "confidence": d.get("ocrConfidence"),
                "id": d.get("_id"),
            }
            for d in documents
        ],
        use_container_width=True,
    )

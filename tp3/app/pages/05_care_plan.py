"""Page 5 - Care Plan."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.shared import page_setup, require_patient, sidebar
from tools import mongodb_tools

page_setup("Care Plan")
sidebar()

st.title("Care plan")
st.caption(
    "Structured output assembled from the multi-agent run. Reasoning, care-plan "
    "generation, and review sections are added in later phases (5-8)."
)

pid = require_patient()
if pid:
    run = mongodb_tools.latest_agent_run(pid)
    if not run:
        st.info("No agent run yet. Launch one from the **Clinical Question** page.")
    else:
        result = run.get("result", {})
        st.write(f"Question: **{run.get('question','')}**")
        if run.get("intent"):
            st.caption(f"Intent: {run['intent']}")

        st.subheader("Identified risks")
        risks = result.get("risks", [])
        if risks:
            st.dataframe(risks, use_container_width=True)
        else:
            st.caption("No risks recorded.")

        if result.get("red_flags"):
            st.subheader("Red flags")
            for f in result["red_flags"]:
                st.error(f)

        st.subheader("Relevant guidelines")
        guidelines = result.get("guidelines", [])
        if guidelines:
            st.dataframe(guidelines, use_container_width=True)
        else:
            st.caption("No guidelines recorded (seed guidelines on the Upload page).")

        st.subheader("Timeline highlights")
        timeline = result.get("timeline", [])
        if timeline:
            highlights = sorted(
                timeline, key=lambda e: e.get("confidence", 0), reverse=True
            )[:5]
            for e in highlights:
                st.write(f"- {e.get('date') or 'Undated'}: {e.get('description','')}")
        else:
            st.caption("No timeline events recorded.")

        st.subheader("Similar cases")
        cases = result.get("similar_cases", [])
        if cases:
            st.dataframe(cases, use_container_width=True)
        else:
            st.caption("No similar cases recorded.")

        st.subheader("Missing information")
        missing = result.get("missing_dates", [])
        if missing:
            for m in missing:
                st.write(f"- {m}")
        else:
            st.caption("None flagged.")

        st.divider()
        st.subheader("Deferred sections (Phases 5-8)")
        for section in [
            "Hypotheses",
            "Supporting / contradicting evidence",
            "Investigation plan",
            "Proposed care plan",
            "Confidence score",
            "Patient-friendly explanation",
            "Clinical review notes",
        ]:
            st.caption(f"• {section} - not yet implemented")

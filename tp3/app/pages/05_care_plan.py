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
        
        # === Step 4 outputs (Amal) ===
        explanation = result.get("patient_explanation") or result.get("patient_friendly_explanation")
        if explanation:
            st.divider()
            st.subheader("Patient-friendly explanation")
            st.write(explanation)

            key_points = result.get("patient_key_points", [])
            if key_points:
                st.markdown("**Key points**")
                for kp in key_points:
                    st.write(f"- {kp}")

            actions = result.get("patient_recommended_actions", [])
            if actions:
                st.markdown("**Recommended actions**")
                for a in actions:
                    st.write(f"- {a}")

            reading = result.get("patient_explanation_reading_level")
            if reading:
                st.caption(f"Estimated reading level: {reading}")

            appropriateness_passed = result.get("patient_appropriateness_passed")
            appropriateness_score = result.get("patient_appropriateness_score")
            if appropriateness_passed is not None or appropriateness_score is not None:
                st.markdown("**Patient representative review**")
                if appropriateness_passed:
                    st.success(f"Appropriateness validated (score: {appropriateness_score})")
                else:
                    st.warning(f"Appropriateness needed revision (score: {appropriateness_score})")
                issues = result.get("patient_appropriateness_issues", [])
                for issue in issues:
                    st.write(f"- {issue}")

        review_assessment = result.get("clinical_review_assessment")
        if review_assessment:
            st.divider()
            st.subheader("Clinical review notes")
            review_passed = result.get("clinical_review_passed")
            clinical_score = result.get("clinical_score")
            if review_passed:
                st.success(f"Clinical review passed (score: {clinical_score})")
            else:
                st.warning(f"Clinical review found issues (score: {clinical_score})")
            st.write(review_assessment)

            missing_safety = result.get("clinical_review_missing_safety_points", [])
            if missing_safety:
                st.markdown("**Missing safety points**")
                for m in missing_safety:
                    st.warning(m)

            unsupported = result.get("clinical_review_unsupported_claims", [])
            if unsupported:
                st.markdown("**Unsupported claims**")
                for u in unsupported:
                    st.write(f"- {u}")

            inconsistencies = result.get("clinical_review_inconsistencies", [])
            if inconsistencies:
                st.markdown("**Inconsistencies**")
                for inc in inconsistencies:
                    st.write(f"- {inc}")

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

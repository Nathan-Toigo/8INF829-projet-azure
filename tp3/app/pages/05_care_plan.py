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
    "Structured output assembled from the multi-agent run including Step 3 "
    "reasoning, care plan, and patient review sections."
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

        st.subheader("Missing timeline dates")
        missing = result.get("missing_dates", [])
        if missing:
            for m in missing:
                st.write(f"- {m}")
        else:
            st.caption("None flagged.")

        st.divider()
        st.subheader("Step 3 — Investigation plan")
        inv_plan = result.get("investigation_plan", [])
        if inv_plan:
            st.dataframe(inv_plan, use_container_width=True)
        else:
            st.caption("No investigation plan recorded.")

        st.subheader("Hypotheses")
        hypotheses = result.get("hypotheses", [])
        if hypotheses:
            st.dataframe(hypotheses, use_container_width=True)
            for note in result.get("hypothesis_rationale", [])[:5]:
                st.caption(note)
        else:
            st.caption("No hypotheses recorded.")

        st.subheader("Evidence")
        evidence = result.get("evidence", [])
        if evidence:
            st.dataframe(evidence, use_container_width=True)
        else:
            st.caption("No evidence recorded.")

        if result.get("contradictions"):
            st.subheader("Contradictions")
            for c in result["contradictions"]:
                st.warning(c)

        if result.get("unsupported_claims"):
            st.subheader("Unsupported claims")
            for u in result["unsupported_claims"]:
                st.write(f"- {u}")

        st.subheader("Clinical gaps")
        gaps = result.get("missing_information", [])
        critical = result.get("critical_gaps", [])
        if gaps or critical:
            if critical:
                st.markdown("**Critical gaps**")
                for g in critical:
                    st.error(g)
            if gaps:
                st.markdown("**Missing information**")
                for g in gaps:
                    st.write(f"- {g}")
        else:
            st.caption("No gaps flagged.")

        st.subheader("Proposed care plan")
        care = result.get("care_plan", [])
        if care:
            st.dataframe(care, use_container_width=True)
        else:
            st.caption("No care plan recorded.")

        conf = result.get("confidence_score")
        if conf is not None:
            st.subheader("Confidence")
            st.metric("Score", f"{conf:.0%}")
            if result.get("confidence_rationale"):
                st.write(result["confidence_rationale"])
            if result.get("step_3_best_confidence"):
                st.caption(f"Best across attempts: {result['step_3_best_confidence']:.0%}")
            if result.get("requires_consensus"):
                st.warning("Low confidence — consensus review recommended.")

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


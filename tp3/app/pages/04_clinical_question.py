"""Page 4 - Clinical Question (launch the multi-agent workflow)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.shared import page_setup, require_patient, sidebar
from config import settings

page_setup("Clinical Question")
sidebar()

st.title("Clinical question")
st.caption("Launch the autonomous multi-agent reasoning workflow.")

EXAMPLES = [
    "What care plan should be proposed?",
    "What risks should be monitored?",
    "What follow-up should the patient discuss with their doctor?",
    "What changed in the last 12 months?",
]

pid = require_patient()
if pid:
    st.write(f"Active patient: `{pid}`")
    example = st.selectbox("Example questions", ["(custom)"] + EXAMPLES)
    default = "" if example == "(custom)" else example
    question = st.text_area("Clinical question", value=default, height=100)

    if not settings.OPENROUTER_API_KEY:
        st.error("OPENROUTER_API_KEY is not set - configure it in .env first.")

    if st.button("Run multi-agent workflow", type="primary", disabled=not question.strip()):
        from graphs import main_graph

        with st.spinner("Agents collaborating (orchestrator -> timeline -> guidelines -> risk -> case investigator)..."):
            try:
                final = main_graph.run_workflow(pid, question.strip())
                st.session_state["last_run_id"] = final.get("run_id")
                st.success(
                    f"Workflow complete in {final.get('step_count', 0)} steps. "
                    f"Agents run: {', '.join(final.get('agents_run', []))}"
                )
                if final.get("errors"):
                    st.warning(f"{len(final['errors'])} error(s) recorded - see Agent Trace.")
                st.info("Open the **Care Plan** and **Agent Trace** pages to review results.")
            except Exception as exc:
                st.error(f"Workflow failed: {exc}")

"""Page 7 - Agent Evaluations (5.3 Reflection Agent logs)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.shared import get_patient_id, page_setup, sidebar
from tools import mongodb_tools

page_setup("Agent Evaluations")
sidebar()

st.title("Agent evaluations")
st.caption(
    "LLM-as-a-Judge reflections: per-run scores for every agent, flagged "
    "underperformers, root causes, and prompt-improvement ideas."
)

pid = get_patient_id()
scope = st.radio(
    "Scope",
    ["All patients", "Active patient only"],
    horizontal=True,
    index=0,
)

try:
    if scope == "Active patient only" and pid:
        evaluations = mongodb_tools.list_agent_evaluations(pid)
    else:
        evaluations = mongodb_tools.list_agent_evaluations()
except Exception:
    st.warning("MongoDB unavailable - start it with `docker compose up -d`.")
    evaluations = []

if not evaluations:
    st.info(
        "No agent evaluations recorded yet. Run a clinical question through the "
        "workflow so the Reflection Agent (5.3) can score the run."
    )
    st.stop()

st.caption(f"{len(evaluations)} evaluation(s) found.")

for ev in evaluations:
    average = ev.get("average", 0.0)
    threshold = ev.get("threshold", 0.0)
    scores = ev.get("scores", []) or []
    flagged = ev.get("flagged", []) or []
    created = ev.get("createdAt", "")

    header = (
        f"{created[:19]}  ·  patient {ev.get('patientId', '?')}  ·  "
        f"avg {average:.2f} / thr {threshold:.2f}  ·  {len(flagged)} flagged"
    )
    with st.expander(header):
        st.markdown(f"**Question:** {ev.get('question', '')}")
        if ev.get("summary"):
            st.markdown(f"**Reflection summary:** {ev['summary']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average", f"{average:.2f}")
        c2.metric("Threshold", f"{threshold:.2f}")
        c3.metric("Agents scored", len(scores))
        c4.metric("Approved", "yes" if ev.get("approved") else "no")

        st.markdown("**Scores by agent**")
        if scores:
            st.dataframe(
                [
                    {
                        "agent": s.get("agent"),
                        "score": s.get("score"),
                        "aligned": s.get("aligned"),
                        "rationale": s.get("rationale"),
                    }
                    for s in scores
                ],
                use_container_width=True,
            )
        else:
            st.caption("No scores recorded.")

        st.markdown("**Flagged agents (below threshold)**")
        if flagged:
            for f in flagged:
                st.markdown(
                    f"- **{f.get('agent')}** (score {f.get('score')})"
                )
                if f.get("rootCause"):
                    st.markdown(f"    - Root cause: {f['rootCause']}")
                improvements = f.get("improvements") or []
                if improvements:
                    st.markdown("    - Improvements:")
                    for imp in improvements:
                        st.markdown(f"        - {imp}")
        else:
            st.caption("No agents were flagged for this run.")

        st.markdown("**Agent final outputs (working memory)**")
        run_id = ev.get("runId")
        run_doc = None
        if run_id:
            try:
                run_doc = mongodb_tools.get_agent_run(run_id)
            except Exception:
                run_doc = None
        outputs = (run_doc or {}).get("agentOutputs", []) or []
        if outputs:
            st.caption(
                f"{len(outputs)} logged step(s) for run {run_id}."
            )
            for entry in outputs:
                icon = {"completed": "✅", "blocked": "⛔", "error": "❌"}.get(
                    entry.get("status"), "•"
                )
                with st.expander(
                    f"step {entry.get('step')}: {icon} {entry.get('agent_id')}"
                ):
                    st.caption(entry.get("handoff_reason", ""))
                    payload = entry.get("output") or {}
                    if payload:
                        st.json(payload)
                    else:
                        st.caption("No memory updates produced by this agent.")
        else:
            st.caption(
                "No agent outputs found for this run "
                "(run the workflow after this update to capture them)."
            )

        baseline = ev.get("baseline")
        if baseline:
            with st.popover("Baseline (final answer used as reference)"):
                st.write(baseline)

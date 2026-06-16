"""Page 3 - Timeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.shared import page_setup, require_patient, sidebar
from tools import mongodb_tools

page_setup("Timeline")
sidebar()

st.title("Patient timeline")
st.caption("Chronological medical history produced by the 2.1 Timeline Agent.")

pid = require_patient()
if pid:
    run = mongodb_tools.latest_agent_run(pid)
    result = (run or {}).get("result", {})
    timeline = result.get("timeline", [])

    if not run:
        st.info(
            "No agent run yet. Go to the **Clinical Question** page to launch the "
            "multi-agent workflow."
        )
    elif not timeline:
        st.warning("The latest run did not produce timeline events.")
    else:
        events = sorted(timeline, key=lambda e: e.get("date") or "9999")
        for e in events:
            conf = e.get("confidence", 0.0)
            flag = "" if e.get("date") else " ⚠️ (date missing/uncertain)"
            st.markdown(
                f"**{e.get('date') or 'Undated'}** - {e.get('description','')}"
                f"  \nConfidence: {conf:.2f} | Source: {e.get('source','') or 'n/a'}{flag}"
            )
            st.divider()

        if result.get("missing_dates"):
            st.subheader("Flagged missing / uncertain dates")
            for m in result["missing_dates"]:
                st.write(f"- {m}")

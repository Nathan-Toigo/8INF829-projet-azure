"""Page 6 - Agent Trace."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.shared import page_setup, require_patient, sidebar
from config import settings
from tools import mongodb_tools

page_setup("Agent Trace")
sidebar()

st.title("Agent trace")
st.caption("Agent execution order, tool calls, token usage, and routing decisions.")

pid = require_patient()
if pid:
    run = mongodb_tools.latest_agent_run(pid)
    if not run:
        st.info("No agent run yet. Launch one from the **Clinical Question** page.")
    else:
        trace = run.get("agentTrace", [])
        tokens = run.get("tokenLedger", [])
        tools = run.get("toolCalls", [])

        c1, c2, c3 = st.columns(3)
        c1.metric("Steps", run.get("stepCount", 0))
        c2.metric("Tool calls", len(tools))
        c3.metric(
            "Total tokens",
            sum(t.get("total_tokens", 0) for t in tokens),
        )

        if settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY:
            st.markdown(
                f"LangSmith project: **{settings.LANGCHAIN_PROJECT}** - "
                "[open LangSmith](https://smith.langchain.com)"
            )
        else:
            st.caption("LangSmith tracing disabled (set LANGCHAIN_* in .env).")

        st.subheader("Execution order & routing decisions")
        for i, entry in enumerate(trace, start=1):
            status = entry.get("status")
            icon = {"completed": "✅", "blocked": "⛔", "error": "❌"}.get(status, "•")
            with st.expander(
                f"{i}. {icon} {entry.get('agent_id')} -> "
                f"{entry.get('next_agent') or ('orchestrator' if entry.get('needs_orchestrator') else 'END')}"
            ):
                st.write(
                    {
                        "status": status,
                        "next_agent": entry.get("next_agent"),
                        "needs_orchestrator": entry.get("needs_orchestrator"),
                        "handoff_reason": entry.get("handoff_reason"),
                        "memory_update_keys": entry.get("memory_update_keys"),
                    }
                )

        st.subheader("Tool calls")
        if tools:
            st.dataframe(
                [
                    {
                        "agent": t.get("step"),
                        "tool": t.get("tool"),
                        "latency_ms": t.get("latency_ms"),
                        "args": str(t.get("args")),
                    }
                    for t in tools
                ],
                use_container_width=True,
            )
        else:
            st.caption("No tool calls recorded.")

        st.subheader("Token usage by step")
        if tokens:
            st.dataframe(tokens, use_container_width=True)
        else:
            st.caption("No token usage recorded.")

        if run.get("errors"):
            st.subheader("Errors & retries")
            st.json(run["errors"])

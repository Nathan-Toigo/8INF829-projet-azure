"""Shared Streamlit helpers: path bootstrap, sidebar, connection status."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config import settings
from tools import chromadb_tools, mongodb_tools


def page_setup(title: str) -> None:
    st.set_page_config(page_title=f"ThreeTokens - {title}", page_icon="🩺", layout="wide")
    settings.configure_langsmith()


def get_patient_id() -> str | None:
    return st.session_state.get("patient_id")


def set_patient_id(patient_id: str | None) -> None:
    st.session_state["patient_id"] = patient_id


def sidebar() -> None:
    """Render the shared sidebar: patient selector + connection status."""
    with st.sidebar:
        st.markdown("### Patient")
        patients = []
        try:
            patients = mongodb_tools.list_patients()
        except Exception:
            st.warning("MongoDB unavailable - start it with `docker compose up -d`.")

        options = ["(none)"] + [p["_id"] for p in patients]
        current = get_patient_id() or "(none)"
        index = options.index(current) if current in options else 0
        chosen = st.selectbox("Active patient", options, index=index)
        set_patient_id(None if chosen == "(none)" else chosen)

        st.divider()
        st.markdown("### System status")
        _status_line("OpenRouter key", bool(settings.OPENROUTER_API_KEY))
        _status_line("MongoDB", mongodb_tools.ping())
        try:
            stats = chromadb_tools.all_stats()
            st.caption(
                "ChromaDB: "
                + ", ".join(f"{k}={v}" for k, v in stats.items())
            )
        except Exception:
            st.caption("ChromaDB: unavailable")
        _status_line(
            "LangSmith tracing",
            settings.LANGCHAIN_TRACING_V2 and bool(settings.LANGCHAIN_API_KEY),
        )


def _status_line(label: str, ok: bool) -> None:
    icon = "✅" if ok else "⚠️"
    st.write(f"{icon} {label}")


def require_patient() -> str | None:
    pid = get_patient_id()
    if not pid:
        st.info("Select or create a patient first (Upload page).")
    return pid

"""ThreeTokens Care Agent - Streamlit entry point.

Run with: ``streamlit run app/main.py`` (from the tp3/ project root).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.shared import page_setup, sidebar
from config import settings
from tools import chromadb_tools, mongodb_tools

page_setup("Home")
sidebar()

st.title("🩺 ThreeTokens Care Agent")
st.caption(
    "Autonomous multi-agent clinical care-plan assistant. "
    "Synthetic data only - not for clinical use."
)

st.markdown(
    """
This app ingests fragmented patient records, indexes them, and runs an
**autonomous multi-agent workflow** (LangGraph + OpenRouter) that builds a
patient timeline, retrieves guidelines, assesses risks, and investigates
similar cases - collaborating through shared **short-term** and cross-agent
**long-term** memory, traced in **LangSmith**.

**Workflow**

1. **Upload** clinical documents -> LLM OCR -> MongoDB + ChromaDB.
2. **Patient Profile / Timeline** - review extracted context.
3. **Clinical Question** - launch the multi-agent reasoning workflow.
4. **Care Plan / Agent Trace** - inspect results and agent collaboration.
"""
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("OpenRouter")
    if settings.OPENROUTER_API_KEY:
        st.success("API key configured")
    else:
        st.error("OPENROUTER_API_KEY missing - set it in .env")
    st.caption(f"Strong: `{settings.OPENROUTER_STRONG_MODEL}`")
    st.caption(f"Small: `{settings.OPENROUTER_SMALL_MODEL}`")
    st.caption(f"Vision: `{settings.OPENROUTER_VISION_MODEL}`")

with col2:
    st.subheader("MongoDB")
    if mongodb_tools.ping():
        st.success(f"Connected ({settings.MONGODB_DB})")
        try:
            mongodb_tools.ensure_indexes()
            counts = {
                c: mongodb_tools.collection(c).estimated_document_count()
                for c in mongodb_tools.COLLECTIONS
            }
            st.caption(", ".join(f"{k}={v}" for k, v in counts.items()))
        except Exception:
            pass
    else:
        st.error("Not reachable - run `docker compose up -d`")

with col3:
    st.subheader("ChromaDB")
    try:
        stats = chromadb_tools.all_stats()
        st.success("Ready")
        for k, v in stats.items():
            st.caption(f"{k}: {v}")
    except Exception as exc:
        st.error(f"Unavailable: {exc}")

st.divider()
st.caption("Use the pages in the left sidebar to navigate the workflow.")

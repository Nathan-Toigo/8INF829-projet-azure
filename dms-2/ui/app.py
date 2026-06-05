"""Streamlit front end for the clinical preparation & coordination agent.

Flow: welcome -> upload/index docs -> run the 5-step agent -> follow-up chat,
with an audit panel exposing every step, tool call, memory write, and token count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from agent import runtime
from agent.mcp_client import tool_result_to_json, tool_result_to_text
from memory.memory_manager import MemoryManager

config.configure_langsmith()


def mcp_call(name: str, args: dict):
    """Call an MCP RAG-admin tool and return the parsed result, or raise."""
    raw = runtime.run_async(runtime.call_tool_async(name, args))
    parsed = tool_result_to_json(raw)
    return parsed if parsed is not None else tool_result_to_text(raw)


def refresh_index_stats() -> dict:
    try:
        stats = mcp_call("index_stats", {})
        if isinstance(stats, dict):
            st.session_state["index_stats"] = stats
            return stats
    except Exception as exc:
        st.session_state["index_stats_error"] = str(exc)
    return st.session_state.get("index_stats", {"chunk_count": 0, "sources": []})

st.set_page_config(page_title="Agent de Preparation Clinique", layout="wide")

LANGUAGES = {"Francais": "fr", "English": "en"}


# --------------------------------------------------------------------- state
def get_memory() -> MemoryManager:
    pid = st.session_state["patient_id"]
    mem = st.session_state.get("memory")
    if mem is None or mem.patient_id != pid:
        mem = MemoryManager(pid)
        st.session_state["memory"] = mem
    return mem


def _init_state() -> None:
    st.session_state.setdefault("patient_id", "demo-patient")
    st.session_state.setdefault("greeting", "")
    st.session_state.setdefault("run_result", None)
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("followup_tokens", [])
    st.session_state.setdefault("index_stats", {"chunk_count": 0, "sources": []})


_init_state()


# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Configuration")
    st.text_input("Patient ID", key="patient_id")
    lang_label = st.selectbox("Langue de la synthese", list(LANGUAGES))
    language = LANGUAGES[lang_label]

    if not config.OPENAI_API_KEY:
        st.error("OPENAI_API_KEY manquant (voir .env).")
    ls_on = config.LANGCHAIN_TRACING_V2 and bool(config.LANGCHAIN_API_KEY)
    st.caption(
        f"LangSmith: {'actif (' + config.LANGCHAIN_PROJECT + ')' if ls_on else 'inactif'}"
    )
    st.caption(f"MCP: {config.MCP_SERVER_URL}")
    st.caption(f"Web search: {'Tavily actif' if config.TAVILY_API_KEY else 'desactive'}")

    st.divider()
    st.subheader("Documents du patient")
    st.caption("Toutes les operations RAG passent par le serveur MCP.")

    if st.button("Reinitialiser l'index"):
        try:
            mcp_call("reset_index", {})
            st.session_state["run_result"] = None
            refresh_index_stats()
            st.success("Index reinitialise.")
        except Exception as exc:
            st.error(f"Echec (serveur MCP demarre ?): {exc}")

    if st.button("Charger le patient exemple (docs/)"):
        with st.spinner("Indexation des documents exemple via MCP..."):
            try:
                res = mcp_call("ingest_directory", {"directory": str(config.DOCS_DIR)})
                refresh_index_stats()
                st.success(
                    f"{res.get('documents', 0)} documents indexes "
                    f"({res.get('chunks_indexed', 0)} chunks)."
                )
            except Exception as exc:
                st.error(f"Echec (serveur MCP demarre ?): {exc}")

    uploads = st.file_uploader(
        "Televerser PDF / DOCX / TXT",
        type=["pdf", "docx", "doc", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Indexer les fichiers televerses"):
        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        total = 0
        try:
            for up in uploads:
                dest = config.UPLOADS_DIR / up.name
                dest.write_bytes(up.getbuffer())
                res = mcp_call("ingest_document", {"file_path": str(dest)})
                total += res.get("chunks_indexed", 0) if isinstance(res, dict) else 0
            refresh_index_stats()
            st.success(f"{len(uploads)} fichiers indexes ({total} chunks).")
        except Exception as exc:
            st.error(f"Echec (serveur MCP demarre ?): {exc}")

    if st.button("Rafraichir l'etat de l'index"):
        refresh_index_stats()

    idx = st.session_state.get("index_stats", {"chunk_count": 0, "sources": []})
    st.metric("Chunks indexes", idx.get("chunk_count", 0))
    if idx.get("sources"):
        st.caption("Sources: " + ", ".join(idx["sources"]))


# ---------------------------------------------------------------------- main
st.title("8INF829 - TP#2 - Agent de Préparation Clinique")
st.caption(
    "Donnees synthetiques uniquement"
)

memory = get_memory()

col_run, col_audit = st.columns([3, 2])

with col_run:
    if st.button("Message de bienvenue de l'agent"):
        with st.spinner("L'agent vous accueille..."):
            text, _tok = runtime.run_async(
                runtime.greet_async(st.session_state["patient_id"], language, memory)
            )
            st.session_state["greeting"] = text
    if st.session_state["greeting"]:
        st.info(st.session_state["greeting"])

    request = st.text_area(
        "Demande / focus (optionnel)",
        placeholder="Ex: Preparer la consultation de suivi, identifier les risques principaux.",
    )

    if st.button("Lancer l'agent (5 etapes)", type="primary"):
        if refresh_index_stats().get("chunk_count", 0) == 0:
            st.warning("Indexez d'abord des documents (barre laterale).")
        else:
            with st.spinner("L'agent execute les 5 etapes cliniques..."):
                result = runtime.run_async(
                    runtime.run_pipeline_async(
                        st.session_state["patient_id"], language, memory, request
                    )
                )
                st.session_state["run_result"] = result

    result = st.session_state["run_result"]
    if result:
        st.subheader("Raisonnement en 5 etapes")
        for step in result.get("step_results", []):
            with st.expander(step["title"], expanded=step["step_id"] == "synthesis"):
                st.markdown(step["output"])
                with st.popover("Sortie structuree"):
                    st.json(step.get("structured", {}))

        st.divider()
        st.subheader("Questions de suivi")
        for q, a in st.session_state["chat"]:
            st.chat_message("user").write(q)
            st.chat_message("assistant").write(a)
        question = st.chat_input("Posez une question de suivi a l'agent")
        if question:
            with st.spinner("Reflexion..."):
                fu = runtime.run_async(
                    runtime.followup_async(
                        st.session_state["patient_id"], language, memory, question
                    )
                )
            st.session_state["chat"].append((question, fu["answer"]))
            st.session_state["followup_tokens"].extend(fu.get("token_records", []))
            st.rerun()


# --------------------------------------------------------------------- audit
with col_audit:
    st.subheader("Panneau d'audit")
    result = st.session_state["run_result"]

    ledger = list(st.session_state["followup_tokens"])
    if result:
        ledger = result.get("token_ledger", []) + ledger

    if ledger:
        st.markdown("**Tokens par etape**")
        per_step: dict[str, dict] = {}
        for rec in ledger:
            agg = per_step.setdefault(
                rec["step"], {"prompt": 0, "completion": 0, "total": 0}
            )
            agg["prompt"] += rec.get("prompt_tokens", 0)
            agg["completion"] += rec.get("completion_tokens", 0)
            agg["total"] += rec.get("total_tokens", 0)
        st.dataframe(
            [{"etape": k, **v} for k, v in per_step.items()],
            use_container_width=True,
            hide_index=True,
        )
        total_tokens = sum(v["total"] for v in per_step.values())
        st.metric("Tokens cumules (agent)", total_tokens)

    if result:
        st.markdown("**Appels d'outils MCP**")
        tool_rows = [
            {
                "etape": tc["step"],
                "outil": tc["tool"],
                "latence_ms": tc["latency_ms"],
            }
            for tc in result.get("tool_calls", [])
        ]
        if tool_rows:
            st.dataframe(tool_rows, use_container_width=True, hide_index=True)
        with st.expander("Details des appels d'outils"):
            st.json(result.get("tool_calls", []))

        st.markdown("**Decisions memoire**")
        st.json(result.get("memory_decisions", []))

    st.markdown("**Etat de la memoire**")
    snap = memory.snapshot()
    st.caption(
        f"Court terme: {snap['short_term_count']} | Long terme: {snap['long_term_count']}"
    )
    with st.expander("Memoire long terme (persistante)"):
        st.json(snap["long_term"])
    with st.expander("Memoire court terme (fenetre glissante)"):
        st.json(snap["short_term"])

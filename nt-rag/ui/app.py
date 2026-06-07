"""Streamlit web UI for the nt-rag conversational agent."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from agent.chat_agent import ChatAgent
from agent import runtime
from agent.mcp_client import tool_result_to_json
from chunking import CHUNK_METHODS
from ollama_client import check_ollama

st.set_page_config(page_title="nt-rag Agent", layout="wide")

CHUNK_LABELS: dict[str, str] = {
    "fixed_chars": "Fixed characters (configurable size/overlap)",
    "paragraph": "Paragraph breaks",
    "page": "One chunk per PDF page",
    "words_250": "250-word sliding windows",
}


def mcp_call(name: str, args: dict):
    result = runtime.run_async(runtime.mcp_call_async(name, args))
    if isinstance(result, (dict, list)):
        return result
    if isinstance(result, str):
        parsed = tool_result_to_json(result)
        if parsed is not None:
            return parsed
        raise RuntimeError(
            f"MCP tool '{name}' returned non-JSON text: {result[:300]}"
        )
    return result


def mcp_call_dict(name: str, args: dict) -> dict:
    result = mcp_call(name, args)
    if isinstance(result, dict):
        return result
    raise RuntimeError(f"MCP tool '{name}' expected dict, got {type(result).__name__}")


def refresh_index_stats(chunk_method: str) -> dict:
    try:
        stats = mcp_call_dict("index_stats", {"chunk_method": chunk_method})
        st.session_state["index_stats"] = stats
        return stats
    except Exception as exc:
        st.session_state["index_stats_error"] = str(exc)
    return st.session_state.get(
        "index_stats",
        {"chunk_count": 0, "sources": [], "chunk_method": chunk_method},
    )


def _sync_agent_chunk_method() -> None:
    agent: ChatAgent = st.session_state["agent"]
    agent.set_chunk_method(st.session_state["active_chunk_method"])


def _set_active_chunk_method(method: str) -> None:
    """Update active chunk method (safe to call from any tab; rerun to refresh sidebar)."""
    st.session_state["active_chunk_method"] = method
    _sync_agent_chunk_method()


def _chunk_method_index(method: str) -> int:
    try:
        return list(CHUNK_METHODS).index(method)
    except ValueError:
        return 0


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("tool_calls", [])
    st.session_state.setdefault("active_chunk_method", "fixed_chars")
    st.session_state.setdefault("index_stats", {"chunk_count": 0, "sources": []})
    st.session_state.setdefault("agent", ChatAgent(chunk_method="fixed_chars"))


_init_state()


with st.sidebar:
    st.header("Configuration")
    st.caption(f"Ollama: {config.OLLAMA_BASE_URL}")
    st.caption(f"Chat model: {config.OLLAMA_CHAT_MODEL}")
    st.caption(f"MCP server: {config.MCP_SERVER_URL}")

    try:
        check_ollama()
        st.success("Ollama reachable")
    except RuntimeError as exc:
        st.error(str(exc))

    try:
        tools = runtime.run_async(runtime.list_mcp_tools_async())
        st.caption(f"MCP tools: {len(tools)} registered")
    except Exception as exc:
        st.warning(f"MCP offline: {exc}")

    st.divider()
    st.subheader("Active chunking")
    active = st.session_state["active_chunk_method"]
    method = st.selectbox(
        "Chunk method for chat & RAG",
        options=list(CHUNK_METHODS),
        format_func=lambda m: CHUNK_LABELS.get(m, m),
        index=_chunk_method_index(active),
    )
    if method != active:
        st.session_state["active_chunk_method"] = method
    _sync_agent_chunk_method()

    idx = refresh_index_stats(method)
    st.metric("Indexed chunks", idx.get("chunk_count", 0))
    st.caption(f"Collection: `{idx.get('collection_name', '?')}`")
    if idx.get("sources"):
        st.caption("Sources: " + ", ".join(idx["sources"][:6]))
        if len(idx["sources"]) > 6:
            st.caption(f"... and {len(idx['sources']) - 6} more")

    st.divider()
    if st.button("New conversation"):
        st.session_state["messages"] = []
        st.session_state["tool_calls"] = []
        st.session_state["agent"] = ChatAgent(
            chunk_method=st.session_state["active_chunk_method"]
        )
        st.rerun()


st.title("Agent conversationnel nt-rag")
st.caption(
    "Chat with MCP tools, choose a chunking strategy, and extend the index with new files."
)

tab_chat, tab_index, tab_upload = st.tabs(
    ["Chat", "Index & chunking", "Add documents"]
)

chunk_method = st.session_state["active_chunk_method"]

# ------------------------------------------------------------------ Chat tab
with tab_chat:
    col_chat, col_audit = st.columns([3, 2])

    with col_chat:
        st.info(
            f"RAG queries use the **{CHUNK_LABELS.get(chunk_method, chunk_method)}** "
            f"index (`{chunk_method}`)."
        )
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask a question...")
        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                try:
                    result = runtime.run_async(
                        runtime.chat_async(st.session_state["agent"], prompt)
                    )
                    answer = result.get("answer", "")
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state["tool_calls"].extend(
                        result.get("tool_calls", [])
                    )
                except Exception as exc:
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": (
                                f"Error: {exc}\n\n"
                                "Check that Ollama and the MCP server are running."
                            ),
                        }
                    )
            st.rerun()

    with col_audit:
        st.subheader("Audit")
        tool_rows = st.session_state.get("tool_calls", [])
        if tool_rows:
            st.markdown("**MCP tool calls**")
            st.dataframe(
                [
                    {"tool": tc["tool"], "latency_ms": tc["latency_ms"]}
                    for tc in tool_rows
                ],
                width="stretch",
                hide_index=True,
            )
            with st.expander("Call details"):
                st.json(tool_rows)
        else:
            st.caption("Tool calls will appear here during the conversation.")

        with st.expander("Index state"):
            st.json(st.session_state.get("index_stats", {}))

# ----------------------------------------------------------- Index & chunking
with tab_index:
    st.subheader("Choose chunking and embed docs/")
    st.markdown(
        "Each chunking strategy uses its own Chroma collection. "
        "Selecting a method and indexing will **chunk, embed, and store** "
        "all PDF/DOCX files from `docs/`."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        index_method = st.selectbox(
            "Chunking method to embed",
            options=list(CHUNK_METHODS),
            format_func=lambda m: CHUNK_LABELS.get(m, m),
            key="index_tab_chunk_method",
        )
    with col_b:
        clear_before = st.checkbox(
            "Clear collection before ingest",
            value=True,
            help="Replace the entire index for this chunk method.",
        )

    if st.button("Embed & index docs/", type="primary"):
        with st.spinner(f"Chunking with `{index_method}` and embedding..."):
            try:
                res = mcp_call_dict(
                    "ingest_documents",
                    {"clear": clear_before, "chunk_method": index_method},
                )
                _set_active_chunk_method(index_method)
                refresh_index_stats(index_method)
                st.success(
                    f"Done: {res.get('documents', '?')} documents, "
                    f"{res.get('chunks', '?')} chunks, "
                    f"{res.get('vectors', '?')} vectors "
                    f"in `{res.get('collection_name', '?')}` "
                    f"({res.get('embed_seconds', '?')}s embed)."
                )
                st.json(res)
                st.rerun()
            except Exception as exc:
                st.error(f"Failed: {exc}")

    st.divider()
    st.subheader("Collections overview")
    rows = []
    for m in CHUNK_METHODS:
        stats = mcp_call_dict("index_stats", {"chunk_method": m})
        rows.append(
            {
                "method": m,
                "label": CHUNK_LABELS.get(m, m),
                "chunks": stats.get("chunk_count", 0),
                "sources": len(stats.get("sources", [])),
                "collection": stats.get("collection_name", ""),
            }
        )
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)

    active = refresh_index_stats(index_method)
    with st.expander(f"Details: {index_method}"):
        st.json(active)

# -------------------------------------------------------------- Add documents
with tab_upload:
    st.subheader("Add new files to the RAG index")
    st.markdown(
        f"Upload PDF or DOCX files. They are saved under `{config.UPLOADS_DIR}` "
        "and indexed into the collection for the selected chunk method."
    )

    upload_method = st.selectbox(
        "Chunking method for uploaded files",
        options=list(CHUNK_METHODS),
        format_func=lambda m: CHUNK_LABELS.get(m, m),
        key="upload_tab_chunk_method",
    )
    replace_existing = st.checkbox(
        "Replace existing chunks for same filename",
        value=True,
        key="upload_replace_existing",
    )

    uploads = st.file_uploader(
        "Upload PDF / DOCX",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
    )

    if uploads and st.button("Save & index uploaded files", type="primary"):
        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []
        try:
            for up in uploads:
                dest = config.UPLOADS_DIR / up.name
                dest.write_bytes(up.getbuffer())
                saved_paths.append(str(dest))

            with st.spinner("Chunking and embedding uploaded files..."):
                res = mcp_call_dict(
                    "ingest_uploaded_files",
                    {
                        "file_paths": saved_paths,
                        "chunk_method": upload_method,
                        "replace_existing": replace_existing,
                    },
                )
            _set_active_chunk_method(upload_method)
            refresh_index_stats(upload_method)
            st.success(
                f"Indexed {len(saved_paths)} file(s): "
                f"{res.get('chunks', '?')} chunks, "
                f"{res.get('vectors', '?')} total vectors in collection."
            )
            st.json(res)
            st.rerun()
        except Exception as exc:
            st.error(f"Failed: {exc}")

    st.divider()
    st.subheader("Files in uploads/")
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        p for p in config.UPLOADS_DIR.iterdir() if p.is_file() and p.name != ".gitkeep"
    )
    if existing:
        st.dataframe(
            [{"file": p.name, "size_kb": round(p.stat().st_size / 1024, 1)} for p in existing],
            width="stretch",
            hide_index=True,
        )

        reindex_paths = st.multiselect(
            "Re-index existing uploads",
            options=[str(p) for p in existing],
            format_func=lambda p: Path(p).name,
        )
        if reindex_paths and st.button("Re-index selected uploads"):
            with st.spinner("Re-indexing..."):
                try:
                    res = mcp_call_dict(
                        "ingest_uploaded_files",
                        {
                            "file_paths": reindex_paths,
                            "chunk_method": upload_method,
                            "replace_existing": replace_existing,
                        },
                    )
                    refresh_index_stats(upload_method)
                    _set_active_chunk_method(upload_method)
                    st.success(f"Re-indexed {len(reindex_paths)} file(s).")
                    st.json(res)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed: {exc}")
    else:
        st.caption("No uploaded files yet.")

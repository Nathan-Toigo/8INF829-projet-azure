"""Page 1 - Patient Upload."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.shared import get_patient_id, page_setup, set_patient_id, sidebar
from config import settings
from ingestion import seed_guidelines, upload_handler
from tools import chromadb_tools

page_setup("Upload")
sidebar()

st.title("Upload patient documents")
st.caption("PDF, image, scanned report, DOCX, or text. Runs LLM OCR via OpenRouter.")

col_a, col_b = st.columns([2, 1])

with col_a:
    patient_id_input = st.text_input(
        "Patient id (leave blank to create a new patient)",
        value=get_patient_id() or "",
    )
    force_vision = st.checkbox(
        "Force vision OCR (use for scanned PDFs even if text is extractable)",
        value=False,
    )
    files = st.file_uploader(
        "Choose one or more files",
        type=["pdf", "png", "jpg", "jpeg", "webp", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Ingest documents", type="primary", disabled=not files):
        pid = patient_id_input.strip() or None
        results = []
        progress = st.progress(0.0)
        for i, f in enumerate(files):
            with st.spinner(f"Processing {f.name} (OCR + extraction + indexing)..."):
                try:
                    summary = upload_handler.ingest_upload(
                        f.getvalue(), f.name, patient_id=pid, force_vision=force_vision
                    )
                    pid = summary["patient_id"]
                    results.append(summary)
                except Exception as exc:
                    st.error(f"{f.name}: {exc}")
            progress.progress((i + 1) / len(files))
        if pid:
            set_patient_id(pid)
            st.success(f"Ingested {len(results)} document(s) for patient `{pid}`.")
        for r in results:
            with st.expander(f"{r['file_name']} - {r['document_type']}"):
                st.write(
                    {
                        "OCR method": r["ocr_method"],
                        "OCR confidence": r["ocr_confidence"],
                        "Pages": r["page_count"],
                        "Clinical resources extracted": r["resources_extracted"],
                        "Chunks indexed": r["chunks_indexed"],
                    }
                )
                st.text_area(
                    "Extracted text preview", r["text_preview"], height=160,
                    key=f"prev-{r['document_id']}",
                )

with col_b:
    st.subheader("Helpers")
    if st.button("Seed clinical guidelines"):
        with st.spinner("Seeding guidelines into ChromaDB..."):
            count = seed_guidelines.seed()
        st.success(f"Seeded {count} guideline snippets.")

    st.caption(f"Sample charts dir: `{settings.DOCS_DIR}`")
    if st.button("Load sample patient from docs/"):
        if not settings.DOCS_DIR.exists():
            st.error(f"Docs dir not found: {settings.DOCS_DIR}")
        else:
            valid = [
                p
                for p in sorted(settings.DOCS_DIR.iterdir())
                if p.suffix.lower() in (".pdf", ".docx", ".txt", ".md")
            ]
            if not valid:
                st.warning("No supported documents found in docs/.")
            else:
                pid = upload_handler.ensure_patient(None, name="Sample patient")
                progress = st.progress(0.0)
                for i, p in enumerate(valid):
                    with st.spinner(f"Ingesting {p.name}..."):
                        try:
                            upload_handler.ingest_path(p, pid)
                        except Exception as exc:
                            st.error(f"{p.name}: {exc}")
                    progress.progress((i + 1) / len(valid))
                set_patient_id(pid)
                st.success(f"Loaded sample patient `{pid}` ({len(valid)} documents).")

    try:
        stats = chromadb_tools.all_stats()
        st.divider()
        st.caption("ChromaDB collections")
        st.json(stats)
    except Exception:
        pass

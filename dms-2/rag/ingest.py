"""Document loaders + chunking for runtime indexing into Chroma.

Extraction ideas (PDF/DOCX page handling, skipping temp/hidden files) are
carried over from the sibling ``dms/document_loader.py``. Chunking uses a simple
word-window with overlap, which is a good default for fragmented clinical notes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PdfReader

from rag import vectorstore

WORDS_PER_CHUNK = 220
WORD_OVERLAP = 40


def _load_pdf(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _load_docx(path: Path) -> list[str]:
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return ["\n\n".join(paragraphs)]


def _load_text(path: Path) -> list[str]:
    return [path.read_text(encoding="utf-8", errors="ignore")]


def load_pages(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in (".docx", ".doc"):
        return _load_docx(path)
    if suffix in (".txt", ".md"):
        return _load_text(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, WORDS_PER_CHUNK - WORD_OVERLAP)
    for start in range(0, len(words), step):
        window = words[start : start + WORDS_PER_CHUNK]
        if window:
            chunks.append(" ".join(window))
        if start + WORDS_PER_CHUNK >= len(words):
            break
    return chunks


def ingest_file(path: Path, source_name: str | None = None) -> dict:
    """Load, chunk, and index a single file into Chroma."""
    path = Path(path)
    source = source_name or path.name
    pages = load_pages(path)
    chunks: list[str] = []
    for page in pages:
        chunks.extend(chunk_text(page))
    added = vectorstore.add_chunks(source, chunks)
    return {"source": source, "pages": len(pages), "chunks_indexed": added}


def ingest_text(text: str, source_name: str) -> dict:
    chunks = chunk_text(text)
    added = vectorstore.add_chunks(source_name, chunks)
    return {"source": source_name, "pages": 1, "chunks_indexed": added}


def _is_valid_doc(path: Path) -> bool:
    name = path.name
    if name.startswith(".") or name.startswith("~$"):
        return False
    return path.suffix.lower() in (".pdf", ".docx", ".doc", ".txt", ".md")


def ingest_directory(docs_dir: Path) -> list[dict]:
    """Index every supported file in a directory (used for the sample patient)."""
    results: list[dict] = []
    for path in sorted(Path(docs_dir).iterdir()):
        if _is_valid_doc(path):
            results.append(ingest_file(path))
    return results

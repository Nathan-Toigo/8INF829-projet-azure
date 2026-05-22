"""Split documents into chunks using multiple strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass

from documents import LoadedDocument

CHUNK_METHODS = ("fixed_chars", "paragraph", "page", "words_250")


@dataclass
class TextChunk:
    chunk_id: str
    source_file: str
    page_index: int
    content: str
    chunk_method: str
    chunk_index: int
    word_count: int = 0


def _word_count(text: str) -> int:
    return len(text.split())


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _make_chunk(
    doc: LoadedDocument,
    method: str,
    chunk_index: int,
    content: str,
    page_index: int = 0,
) -> TextChunk:
    return TextChunk(
        chunk_id=f"{doc.source_file}::{method}::p{page_index}::c{chunk_index}",
        source_file=doc.source_file,
        page_index=page_index,
        content=content,
        chunk_method=method,
        chunk_index=chunk_index,
        word_count=_word_count(content),
    )


def chunk_fixed_chars(
    documents: list[LoadedDocument],
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for doc in documents:
        for page_idx, page_text in enumerate(doc.pages):
            if not page_text.strip():
                continue
            for part_idx, part in enumerate(_split_text(page_text, chunk_size, overlap)):
                chunks.append(
                    _make_chunk(doc, "fixed_chars", part_idx, part, page_idx)
                )
    return chunks


def chunk_paragraph(doc: LoadedDocument) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    idx = 0
    for page_idx, page_text in enumerate(doc.pages):
        if not page_text.strip():
            continue
        parts = re.split(r"\n\s*\n+", page_text)
        for part in parts:
            part = part.strip()
            if len(part) < 40:
                continue
            chunks.append(_make_chunk(doc, "paragraph", idx, part, page_idx))
            idx += 1
    return chunks


def chunk_page(doc: LoadedDocument) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page_idx, page_text in enumerate(doc.pages):
        page_text = page_text.strip()
        if len(page_text) < 30:
            continue
        chunks.append(_make_chunk(doc, "page", page_idx, page_text, page_idx))
    return chunks


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    segments: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        segment = " ".join(words[start:end])
        if segment.strip():
            segments.append(segment)
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return segments


def chunk_words_250(doc: LoadedDocument) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    idx = 0
    full_text = "\n\n".join(p for p in doc.pages if p.strip())
    for segment in _split_words(full_text, size=250, overlap=25):
        chunks.append(_make_chunk(doc, "words_250", idx, segment, 0))
        idx += 1
    return chunks


def chunk_documents(
    documents: list[LoadedDocument],
    method: str = "fixed_chars",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    if method == "fixed_chars":
        return chunk_fixed_chars(documents, chunk_size, chunk_overlap)
    if method == "paragraph":
        out: list[TextChunk] = []
        for doc in documents:
            out.extend(chunk_paragraph(doc))
        return out
    if method == "page":
        out = []
        for doc in documents:
            out.extend(chunk_page(doc))
        return out
    if method == "words_250":
        out = []
        for doc in documents:
            out.extend(chunk_words_250(doc))
        return out
    raise ValueError(f"Unknown chunk method: {method}. Use: {CHUNK_METHODS}")

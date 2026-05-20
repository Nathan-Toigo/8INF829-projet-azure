"""Comparative chunking strategies for RAG ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass

from document_loader import LoadedDocument


@dataclass
class TextChunk:
    source_file: str
    chunk_method: str
    chunk_index: int
    content: str
    page_number: int | None = None
    word_count: int = 0


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_paragraph(doc: LoadedDocument) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    idx = 0
    for page_num, page_text in enumerate(doc.pages, start=1):
        if not page_text.strip():
            continue
        parts = re.split(r"\n\s*\n+", page_text)
        for part in parts:
            part = part.strip()
            if len(part) < 40:
                continue
            chunks.append(
                TextChunk(
                    source_file=doc.source_file,
                    chunk_method="paragraph",
                    chunk_index=idx,
                    content=part,
                    page_number=page_num,
                    word_count=_word_count(part),
                )
            )
            idx += 1
    return chunks


def chunk_page(doc: LoadedDocument) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page_num, page_text in enumerate(doc.pages, start=1):
        page_text = page_text.strip()
        if len(page_text) < 30:
            continue
        chunks.append(
            TextChunk(
                source_file=doc.source_file,
                chunk_method="page",
                chunk_index=page_num - 1,
                content=page_text,
                page_number=page_num,
                word_count=_word_count(page_text),
            )
        )
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
        chunks.append(
            TextChunk(
                source_file=doc.source_file,
                chunk_method="words_250",
                chunk_index=idx,
                content=segment,
                word_count=_word_count(segment),
            )
        )
        idx += 1
    return chunks


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_llm_optimized(doc: LoadedDocument) -> list[TextChunk]:
    """Sentence-aware chunks ~400 words with 50-word overlap."""
    target_words = 400
    overlap_words = 50
    chunks: list[TextChunk] = []
    idx = 0
    full_text = "\n\n".join(p for p in doc.pages if p.strip())
    sentences = _split_sentences(full_text)
    if not sentences:
        return chunks

    current: list[str] = []
    current_wc = 0
    for sentence in sentences:
        sw = _word_count(sentence)
        if current_wc + sw > target_words and current:
            content = " ".join(current)
            chunks.append(
                TextChunk(
                    source_file=doc.source_file,
                    chunk_method="llm_optimized",
                    chunk_index=idx,
                    content=content,
                    word_count=_word_count(content),
                )
            )
            idx += 1
            overlap = []
            overlap_wc = 0
            for s in reversed(current):
                overlap.insert(0, s)
                overlap_wc += _word_count(s)
                if overlap_wc >= overlap_words:
                    break
            current = overlap + [sentence]
            current_wc = sum(_word_count(s) for s in current)
        else:
            current.append(sentence)
            current_wc += sw

    if current:
        content = " ".join(current)
        if len(content) >= 40:
            chunks.append(
                TextChunk(
                    source_file=doc.source_file,
                    chunk_method="llm_optimized",
                    chunk_index=idx,
                    content=content,
                    word_count=_word_count(content),
                )
            )
    return chunks


CHUNKERS = {
    "paragraph": chunk_paragraph,
    "page": chunk_page,
    "words_250": chunk_words_250,
    "llm_optimized": chunk_llm_optimized,
}


def chunk_document(doc: LoadedDocument, method: str) -> list[TextChunk]:
    if method not in CHUNKERS:
        raise ValueError(f"Unknown chunk method: {method}")
    return CHUNKERS[method](doc)


def chunk_all_methods(doc: LoadedDocument) -> dict[str, list[TextChunk]]:
    return {method: chunk_document(doc, method) for method in CHUNKERS}

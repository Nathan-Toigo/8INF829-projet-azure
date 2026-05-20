"""Split documents into overlapping text chunks for embedding."""

from __future__ import annotations

from dataclasses import dataclass

from documents import LoadedDocument


@dataclass
class TextChunk:
    chunk_id: str
    source_file: str
    page_index: int
    content: str


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


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for doc in documents:
        for page_idx, page_text in enumerate(doc.pages):
            if not page_text.strip():
                continue
            for part_idx, part in enumerate(
                _split_text(page_text, chunk_size, overlap)
            ):
                chunk_id = f"{doc.source_file}::p{page_idx}::c{part_idx}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        source_file=doc.source_file,
                        page_index=page_idx,
                        content=part,
                    )
                )
    return chunks

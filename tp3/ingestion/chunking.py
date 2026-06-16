"""Word-window chunking for indexing clinical text into ChromaDB.

A simple word-window with overlap is a robust default for fragmented clinical
notes (carried over from the sibling ``dms-2/rag/ingest.py``).
"""

from __future__ import annotations

WORDS_PER_CHUNK = 220
WORD_OVERLAP = 40


def chunk_text(text: str) -> list[str]:
    words = (text or "").split()
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

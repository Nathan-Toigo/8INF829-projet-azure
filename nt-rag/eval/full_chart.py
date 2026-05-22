"""Full-document baseline answers for benchmark comparison."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import config
from documents import build_full_corpus
from eval.prompts import PHYSICIAN_SYSTEM_PROMPT, USER_PROMPT_FULL_DOCS_TEMPLATE
from ollama_client import chat_completion

_corpus_cache: tuple[str, dict] | None = None


def get_full_corpus(docs_dir: Path | None = None) -> tuple[str, dict]:
    global _corpus_cache
    if _corpus_cache is None:
        _corpus_cache = build_full_corpus(
            docs_dir or config.DOCS_DIR,
            max_chars=config.MAX_FULL_DOC_CHARS,
        )
    return _corpus_cache


def answer_with_full_documents(
    question: str,
    *,
    chat_model: str | None = None,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    corpus, meta = get_full_corpus(docs_dir)
    chat = chat_model or config.OLLAMA_CHAT_MODEL
    t0 = time.perf_counter()
    answer = chat_completion(
        [
            {"role": "system", "content": PHYSICIAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_FULL_DOCS_TEMPLATE.format(
                    question=question,
                    context=corpus,
                ),
            },
        ],
        model=chat,
        temperature=0.2,
    )
    elapsed = time.perf_counter() - t0
    return {
        "question": question,
        "answer": answer,
        "answer_mode": "full_documents",
        "chat_model": chat,
        "corpus_meta": meta,
        "metrics": {"full_chart_sec": round(elapsed, 3)},
    }

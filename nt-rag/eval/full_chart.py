"""Full-document baseline answers for benchmark comparison."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import config
from documents import build_full_corpus
from eval.prompts import PHYSICIAN_SYSTEM_PROMPT, USER_PROMPT_FULL_DOCS_TEMPLATE
from ollama_client import chat_completion

_corpus_cache: dict[int, tuple[str, dict]] = {}


def get_full_corpus(
    docs_dir: Path | None = None,
    max_chars: int | None = None,
) -> tuple[str, dict]:
    limit = max_chars if max_chars is not None else config.MAX_FULL_DOC_CHARS
    if limit not in _corpus_cache:
        _corpus_cache[limit] = build_full_corpus(
            docs_dir or config.DOCS_DIR,
            max_chars=limit,
        )
    return _corpus_cache[limit]


def answer_with_full_documents(
    question: str,
    *,
    chat_model: str | None = None,
    docs_dir: Path | None = None,
    max_chars: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    corpus, meta = get_full_corpus(docs_dir, max_chars=max_chars)
    chat = chat_model or config.OLLAMA_CHAT_MODEL
    prompt_chars = len(corpus) + len(question)
    cap = meta.get("truncated_chars") or meta.get("total_chars")
    truncated = meta.get("truncated", False)
    if verbose:
        total_c = meta.get("total_chars") or 0
        print(
            f"      corpus: {meta.get('document_count')} docs, "
            f"{total_c:,} chars total"
            f"{', truncated' if truncated else ''} (cap {cap})",
            flush=True,
        )
        print(
            f"      prompt ~{prompt_chars:,} chars, model={chat} "
            f"(generation may take several minutes)...",
            flush=True,
        )
    else:
        print(
            f"  Full-chart prompt ~{prompt_chars:,} chars (corpus cap {cap})",
            flush=True,
        )
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
        timeout=config.OLLAMA_CHAT_TIMEOUT,
    )
    elapsed = time.perf_counter() - t0
    return {
        "question": question,
        "answer": answer,
        "answer_mode": "full_documents",
        "chat_model": chat,
        "corpus_meta": meta,
        "metrics": {
            "full_chart_sec": round(elapsed, 3),
            "prompt_chars": prompt_chars,
        },
    }

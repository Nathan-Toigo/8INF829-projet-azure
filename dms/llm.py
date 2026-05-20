#!/usr/bin/env python3
"""LLM initialization, RAG query, and physician-style answering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import time
from dataclasses import dataclass, asdict
from typing import Any

from openai import OpenAI

import config
from database import init_schema, search_similar
from ingest import embed_texts, get_openai_client
from document_loader import build_full_corpus
from prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    PHYSICIAN_SYSTEM_PROMPT,
    USER_PROMPT_FULL_DOCS_TEMPLATE,
    USER_PROMPT_TEMPLATE,
)


@dataclass
class RetrievedChunk:
    rank: int
    source_file: str
    chunk_method: str
    chunk_index: int
    similarity: float
    distance: float
    word_count: int | None
    content_preview: str
    embedding_model: str
    page_number: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RagQueryResult:
    query: str
    embedding_model: str
    candidates: list[RetrievedChunk]
    top_chunks: list[RetrievedChunk]
    per_method_counts: dict[str, int]
    query_embed_ms: float

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "embedding_model": self.embedding_model,
            "per_method_counts": self.per_method_counts,
            "query_embed_ms": self.query_embed_ms,
            "candidates": [c.to_dict() for c in self.candidates],
            "top_chunks": [c.to_dict() for c in self.top_chunks],
        }


_llm_client: OpenAI | None = None
_corpus_cache: tuple[str, dict] | None = None


def init_llm() -> OpenAI:
    """Initialize OpenAI client for chat completions."""
    global _llm_client
    _llm_client = get_openai_client()
    init_schema()
    return _llm_client


def get_llm_client() -> OpenAI:
    if _llm_client is None:
        return init_llm()
    return _llm_client


def rag_query(
    query: str,
    embedding_model: str | None = None,
    top_k_per_method: int | None = None,
    top_n: int | None = None,
) -> RagQueryResult:
    """
    1. Embed search string
    2. Top-K per chunk method (default 5)
    3. Merge and sort by similarity
    4. Return metrics per chunk/method
    5. Select top N chunks (default 2) for prompt injection
    """
    client = get_llm_client()
    model = embedding_model or config.OPENAI_EMBEDDING_MODEL
    k = top_k_per_method or config.TOP_K_PER_METHOD
    n = top_n or config.TOP_CHUNKS_FOR_PROMPT

    t0 = time.perf_counter()
    query_vec = embed_texts(client, [query], model)[0]
    embed_ms = (time.perf_counter() - t0) * 1000

    seen_ids: set[tuple] = set()
    candidates: list[RetrievedChunk] = []
    per_method: dict[str, int] = {}

    for method in config.CHUNK_METHODS:
        hits = search_similar(query_vec, model, chunk_method=method, limit=k)
        per_method[method] = len(hits)
        for hit in hits:
            key = (hit["source_file"], hit["chunk_method"], hit["chunk_index"])
            if key in seen_ids:
                continue
            seen_ids.add(key)
            preview = hit["content"][:400] + ("..." if len(hit["content"]) > 400 else "")
            candidates.append(
                RetrievedChunk(
                    rank=0,
                    source_file=hit["source_file"],
                    chunk_method=hit["chunk_method"],
                    chunk_index=hit["chunk_index"],
                    similarity=float(hit["similarity"]),
                    distance=float(hit["distance"]),
                    word_count=hit.get("word_count"),
                    content_preview=preview,
                    embedding_model=hit["embedding_model"],
                    page_number=hit.get("page_number"),
                )
            )

    candidates.sort(key=lambda c: c.similarity, reverse=True)
    for i, c in enumerate(candidates, start=1):
        c.rank = i

    top_chunks = candidates[:n]
    return RagQueryResult(
        query=query,
        embedding_model=model,
        candidates=candidates,
        top_chunks=top_chunks,
        per_method_counts=per_method,
        query_embed_ms=round(embed_ms, 2),
    )


def format_context(chunks: list[RetrievedChunk], full_texts: dict | None = None) -> str:
    """Build context block from top chunks for the physician prompt."""
    parts = []
    for i, ch in enumerate(chunks, start=1):
        body = full_texts.get((ch.source_file, ch.chunk_method, ch.chunk_index)) if full_texts else None
        if not body:
            body = ch.content_preview
        parts.append(
            f"### Excerpt {i} (method={ch.chunk_method}, file={ch.source_file}, "
            f"similarity={ch.similarity:.4f}, distance={ch.distance:.4f})\n{body}"
        )
    return "\n\n".join(parts)


def get_full_document_corpus() -> tuple[str, dict]:
    """Load and cache the full document corpus for full-chart answering."""
    global _corpus_cache
    if _corpus_cache is None:
        _corpus_cache = build_full_corpus(
            config.DOCS_DIR, max_chars=config.MAX_FULL_DOC_CHARS
        )
    return _corpus_cache


def _call_physician_llm(
    question: str,
    context: str,
    template: str,
    chat_model: str | None = None,
) -> dict[str, Any]:
    client = get_llm_client()
    model = chat_model or config.OPENAI_CHAT_MODEL
    user_message = template.format(question=question, context=context)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PHYSICIAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    llm_ms = (time.perf_counter() - t0) * 1000
    usage = response.usage
    return {
        "answer": response.choices[0].message.content or "",
        "chat_model": model,
        "metrics": {
            "llm_latency_ms": round(llm_ms, 2),
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        },
    }


def fetch_full_chunk_texts(result: RagQueryResult) -> dict[tuple, str]:
    """Load full chunk text from DB for prompt injection."""
    from database import get_connection

    keys = {(c.source_file, c.chunk_method, c.chunk_index) for c in result.top_chunks}
    if not keys:
        return {}
    out: dict[tuple, str] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            for sf, method, idx in keys:
                cur.execute(
                    """
                    SELECT content FROM document_chunks
                    WHERE source_file = %s AND chunk_method = %s
                      AND chunk_index = %s AND embedding_model = %s
                    LIMIT 1
                    """,
                    (sf, method, idx, result.embedding_model),
                )
                row = cur.fetchone()
                if row:
                    out[(sf, method, idx)] = row[0]
    return out


def answer_with_rag(
    question: str,
    embedding_model: str | None = None,
    chat_model: str | None = None,
) -> dict[str, Any]:
    """Full RAG + LLM answer with metrics."""
    rag = rag_query(question, embedding_model=embedding_model)
    full_texts = fetch_full_chunk_texts(rag)
    context = format_context(rag.top_chunks, full_texts)

    llm_result = _call_physician_llm(
        question, context, USER_PROMPT_TEMPLATE, chat_model=chat_model
    )
    return {
        "question": question,
        "answer": llm_result["answer"],
        "answer_mode": "rag",
        "chat_model": llm_result["chat_model"],
        "rag": rag.to_dict(),
        "rag_context": context,
        "metrics": {
            "query_embed_ms": rag.query_embed_ms,
            **llm_result["metrics"],
        },
    }


def answer_with_full_documents(
    question: str,
    chat_model: str | None = None,
) -> dict[str, Any]:
    """Same physician LLM with the full document corpus injected into the prompt."""
    corpus, corpus_meta = get_full_document_corpus()
    llm_result = _call_physician_llm(
        question,
        corpus,
        USER_PROMPT_FULL_DOCS_TEMPLATE,
        chat_model=chat_model,
    )
    return {
        "question": question,
        "answer": llm_result["answer"],
        "answer_mode": "full_documents",
        "chat_model": llm_result["chat_model"],
        "corpus_meta": corpus_meta,
        "metrics": llm_result["metrics"],
    }


def judge_rag_vs_full(
    question: str,
    rag_answer: str,
    full_doc_answer: str,
    rag_context: str,
    judge_model: str | None = None,
) -> dict[str, Any]:
    """Judge LLM: compare RAG answer vs full-chart answer; return scores."""
    client = get_llm_client()
    model = judge_model or config.OPENAI_JUDGE_MODEL
    user_message = JUDGE_USER_TEMPLATE.format(
        question=question,
        rag_context=rag_context,
        rag_answer=rag_answer,
        full_doc_answer=full_doc_answer,
    )

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    llm_ms = (time.perf_counter() - t0) * 1000
    raw = response.choices[0].message.content or "{}"
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {
            "accuracy_score": None,
            "quality_score": None,
            "thoroughness_score": None,
            "global_accuracy_score": None,
            "parse_error": raw,
        }

    usage = response.usage
    return {
        "judge_model": model,
        "scores": scores,
        "metrics": {
            "llm_latency_ms": round(llm_ms, 2),
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        },
    }


def _tokens(metrics: dict) -> int:
    val = metrics.get("total_tokens")
    return int(val) if val is not None else 0


def compute_question_token_usage(
    rag_result: dict,
    full_result: dict,
    judge_result: dict,
) -> dict[str, Any]:
    """
    Compare token usage: RAG vs Full-Chart vs Judge for one question.
    Full-Chart additional = full_chart total minus RAG total (absolute and % vs RAG).
    """
    rag_m = rag_result.get("metrics", {})
    full_m = full_result.get("metrics", {})
    judge_m = judge_result.get("metrics", {})

    rag_total = _tokens(rag_m)
    full_total = _tokens(full_m)
    judge_total = _tokens(judge_m)
    question_total = rag_total + full_total + judge_total

    additional = full_total - rag_total
    if rag_total > 0:
        additional_pct_vs_rag = round((additional / rag_total) * 100, 1)
    else:
        additional_pct_vs_rag = None

    if question_total > 0:
        full_pct_of_question = round((full_total / question_total) * 100, 1)
        rag_pct_of_question = round((rag_total / question_total) * 100, 1)
    else:
        full_pct_of_question = None
        rag_pct_of_question = None

    return {
        "rag_tokens": rag_total,
        "full_chart_tokens": full_total,
        "judge_tokens": judge_total,
        "question_total_tokens": question_total,
        "full_chart_additional_tokens": additional,
        "full_chart_additional_pct_vs_rag": additional_pct_vs_rag,
        "full_chart_pct_of_question_tokens": full_pct_of_question,
        "rag_pct_of_question_tokens": rag_pct_of_question,
        "rag_prompt_tokens": rag_m.get("prompt_tokens"),
        "rag_completion_tokens": rag_m.get("completion_tokens"),
        "full_chart_prompt_tokens": full_m.get("prompt_tokens"),
        "full_chart_completion_tokens": full_m.get("completion_tokens"),
        "judge_prompt_tokens": judge_m.get("prompt_tokens"),
        "judge_completion_tokens": judge_m.get("completion_tokens"),
    }


def evaluate_question(
    qid: str,
    question: str,
    embedding_model: str | None = None,
    chat_model: str | None = None,
) -> dict[str, Any]:
    """Run RAG answer, full-doc answer, and judge comparison for one question."""
    rag_result = answer_with_rag(
        question, embedding_model=embedding_model, chat_model=chat_model
    )
    full_result = answer_with_full_documents(question, chat_model=chat_model)
    judge_result = judge_rag_vs_full(
        question=question,
        rag_answer=rag_result["answer"],
        full_doc_answer=full_result["answer"],
        rag_context=rag_result.get("rag_context", ""),
    )
    token_usage = compute_question_token_usage(rag_result, full_result, judge_result)
    return {
        "id": qid,
        "question": question,
        "rag": rag_result,
        "full_documents": full_result,
        "judge": judge_result,
        "token_usage": token_usage,
    }


def print_rag_metrics(rag: RagQueryResult) -> None:
    print_rag_metrics_from_dict(rag.to_dict())


def print_rag_metrics_from_dict(rag: dict) -> None:
    print("\n  --- RAG retrieval metrics ---")
    print(f"  Embedding model: {rag['embedding_model']}")
    print(f"  Query embed time: {rag['query_embed_ms']} ms")
    print(f"  Hits per method: {rag['per_method_counts']}")
    print(f"  Total candidates (deduped): {len(rag['candidates'])}")
    print("\n  All candidates (sorted by similarity):")
    for c in rag["candidates"][:15]:
        print(
            f"    #{c['rank']} [{c['chunk_method']}] {c['source_file']} "
            f"idx={c['chunk_index']} sim={c['similarity']:.4f} dist={c['distance']:.4f} "
            f"words={c['word_count']}"
        )
    print("\n  Top chunks selected for prompt:")
    for c in rag["top_chunks"]:
        print(
            f"    [{c['chunk_method']}] {c['source_file']} sim={c['similarity']:.4f}"
        )


def main() -> int:
    if not config.OPENAI_API_KEY:
        print("Set OPENAI_API_KEY before running llm.py", file=sys.stderr)
        return 1
    init_llm()
    q = (
        "What was the FNA pathology interpretation for the left cervical lymph node?"
        if len(sys.argv) < 2
        else " ".join(sys.argv[1:])
    )
    result = answer_with_rag(q)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

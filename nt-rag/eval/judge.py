"""LLM judge for RAG answer quality (Ollama)."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import config
from eval.prompts import (
    JUDGE_GOLDEN_TEMPLATE,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
)
from ollama_client import chat_completion


def _parse_json_from_llm(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"parse_error": raw}


def judge_rag_vs_full(
    question: str,
    rag_answer: str,
    full_doc_answer: str,
    rag_context: str,
    *,
    judge_model: str | None = None,
) -> dict[str, Any]:
    model = judge_model or config.OLLAMA_JUDGE_MODEL
    user_message = JUDGE_USER_TEMPLATE.format(
        question=question,
        rag_context=rag_context,
        rag_answer=rag_answer,
        full_doc_answer=full_doc_answer,
    )
    t0 = time.perf_counter()
    raw = chat_completion(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        model=model,
        temperature=0.1,
        timeout=config.OLLAMA_CHAT_TIMEOUT,
    )
    judge_sec = time.perf_counter() - t0
    scores = _parse_json_from_llm(raw)
    return {
        "judge_model": model,
        "scores": scores,
        "metrics": {"judge_sec": round(judge_sec, 3)},
    }


def judge_vs_golden(
    question: str,
    rag_answer: str,
    golden_answer: str,
    rag_context: str,
    *,
    judge_model: str | None = None,
) -> dict[str, Any]:
    model = judge_model or config.OLLAMA_JUDGE_MODEL
    user_message = JUDGE_GOLDEN_TEMPLATE.format(
        question=question,
        golden_answer=golden_answer,
        rag_answer=rag_answer,
        rag_context=rag_context,
    )
    t0 = time.perf_counter()
    raw = chat_completion(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        model=model,
        temperature=0.1,
        timeout=config.OLLAMA_CHAT_TIMEOUT,
    )
    judge_sec = time.perf_counter() - t0
    scores = _parse_json_from_llm(raw)
    return {
        "judge_model": model,
        "scores": scores,
        "metrics": {"judge_sec": round(judge_sec, 3)},
    }

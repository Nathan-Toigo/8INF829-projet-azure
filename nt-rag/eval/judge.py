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

SCORE_KEYS = (
    "accuracy_score",
    "quality_score",
    "thoroughness_score",
    "global_accuracy_score",
    "golden_match_score",
)

_SCORE_RE = re.compile(
    r'"(accuracy_score|quality_score|thoroughness_score|'
    r'global_accuracy_score|golden_match_score)"\s*:\s*(\d{1,3})',
    re.IGNORECASE,
)


def _strip_markdown_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)
    return raw.strip()


def _extract_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _scores_from_regex(text: str) -> dict[str, int]:
    found: dict[str, int] = {}
    for match in _SCORE_RE.finditer(text):
        key = match.group(1)
        val = int(match.group(2))
        if 0 <= val <= 100:
            found[key] = val
    return found


def _normalize_score_dict(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in SCORE_KEYS:
        if key not in data:
            continue
        val = data[key]
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            iv = int(round(val))
            if 0 <= iv <= 100:
                out[key] = iv
    return out


def _parse_json_from_llm(raw: str) -> dict[str, Any]:
    """Parse judge JSON; fall back to regex scores if JSON is truncated."""
    cleaned = _strip_markdown_fence(raw)

    for candidate in (cleaned, _extract_balanced_json(cleaned) or ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            scores = _normalize_score_dict(parsed)
            if scores:
                return scores

    regex_scores = _scores_from_regex(cleaned)
    if regex_scores:
        regex_scores["_parse_partial"] = True
        return regex_scores

    return {"parse_error": cleaned[:2000]}


def _judge_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
) -> str:
    return chat_completion(
        messages,
        model=model,
        temperature=0.0,
        timeout=config.OLLAMA_CHAT_TIMEOUT,
        num_predict=256,
    )


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
    raw = _judge_chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        model=model,
    )
    judge_sec = time.perf_counter() - t0
    scores = _parse_json_from_llm(raw)
    return {
        "judge_model": model,
        "scores": scores,
        "metrics": {"judge_sec": round(judge_sec, 3)},
        "raw_preview": raw[:300] if scores.get("parse_error") else None,
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
    raw = _judge_chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        model=model,
    )
    judge_sec = time.perf_counter() - t0
    scores = _parse_json_from_llm(raw)
    return {
        "judge_model": model,
        "scores": scores,
        "metrics": {"judge_sec": round(judge_sec, 3)},
        "raw_preview": raw[:300] if scores.get("parse_error") else None,
    }

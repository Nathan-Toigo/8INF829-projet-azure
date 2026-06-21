"""OpenRouter (ChatOpenAI) wrapper with per-call token usage capture.

A single OpenRouter key powers every model tier. ``ChatOpenAI`` is pointed at the
OpenRouter base URL so calls are natively traced by LangSmith and tools can be
bound directly. Every call returns a token-usage record callers append to the
short-term-memory token ledger.

Tiers (spec section 14):
- ``strong`` -> clinical review, confidence, consensus, care planning.
- ``small``  -> rewriting, classification, formatting, OCR cleanup.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import settings

_models: dict[tuple[str, float], ChatOpenAI] = {}

_TIER_MODELS = {
    "strong": settings.OPENROUTER_STRONG_MODEL,
    "small": settings.OPENROUTER_SMALL_MODEL,
    "vision": settings.OPENROUTER_VISION_MODEL,
}


def model_name(tier: str, model_override: str | None = None) -> str:
    if model_override:
        return model_override
    return _TIER_MODELS.get(tier, settings.OPENROUTER_SMALL_MODEL)


def get_model(
    tier: str = "small",
    temperature: float = 0.2,
    model_override: str | None = None,
) -> ChatOpenAI:
    resolved = model_name(tier, model_override)
    key = (resolved, temperature)
    if key not in _models:
        _models[key] = ChatOpenAI(
            model=resolved,
            temperature=temperature,
            api_key=settings.require_openrouter_key(),
            base_url=settings.OPENROUTER_BASE_URL,
            default_headers=settings.openrouter_default_headers(),
        )
    return _models[key]


def _usage_record(step: str, tier: str, message: Any, model_override: str | None = None) -> dict:
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "step": step,
        "model": model_name(tier, model_override),
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def invoke_text(
    step: str,
    system: str,
    user: str,
    tier: str = "small",
    temperature: float = 0.3,
) -> tuple[str, dict]:
    model = get_model(tier, temperature)
    message = model.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return (message.content or ""), _usage_record(step, tier, message)


def invoke_structured(
    step: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    tier: str = "small",
    temperature: float = 0.2,
    model_override: str | None = None,
) -> tuple[BaseModel, dict]:
    model = get_model(tier, temperature, model_override=model_override)
    structured = model.with_structured_output(schema, include_raw=True)
    result = structured.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    parsed = result.get("parsed")
    if parsed is None:  # fall back to an empty instance on parse failure
        parsed = schema()
    return parsed, _usage_record(step, tier, result.get("raw"), model_override)


def _result_to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    content = getattr(result, "content", None)
    if content is not None:
        return content if isinstance(content, str) else str(content)
    return str(result)


def run_agentic_step(
    step: str,
    system: str,
    user: str,
    tools: list,
    schema: type[BaseModel],
    tier: str = "small",
    max_iters: int = 4,
    temperature: float = 0.2,
    model_override: str | None = None,
) -> tuple[BaseModel, list[dict], list[dict]]:
    """ReAct-style step: bind tools to the model, let it call them iteratively to
    enrich the memory-seeded context, then emit a structured result.

    Returns ``(parsed_schema, tool_call_records, token_records)``.
    """
    model = get_model(tier, temperature, model_override=model_override)
    by_name = {t.name: t for t in tools}
    messages: list[BaseMessage] = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    tool_records: list[dict] = []
    token_records: list[dict] = []

    if tools:
        llm_with_tools = model.bind_tools(tools)
        for _ in range(max_iters):
            ai = llm_with_tools.invoke(messages)
            token_records.append(_usage_record(step, tier, ai, model_override))
            messages.append(ai)
            calls = getattr(ai, "tool_calls", None) or []
            if not calls:
                break
            for tc in calls:
                name = tc.get("name")
                args = tc.get("args", {}) or {}
                call_id = tc.get("id")
                tool = by_name.get(name)
                t0 = time.perf_counter()
                if tool is None:
                    result_text = f"[tool '{name}' unavailable]"
                else:
                    try:
                        result = tool.invoke(args)
                        result_text = _result_to_text(result)
                    except Exception as exc:  # pragma: no cover - runtime guard
                        result_text = f"[tool '{name}' error: {exc}]"
                ms = (time.perf_counter() - t0) * 1000
                tool_records.append(
                    {
                        "step": step,
                        "tool": name,
                        "args": args,
                        "latency_ms": round(ms, 2),
                        "result_preview": result_text[:600],
                    }
                )
                messages.append(
                    ToolMessage(content=result_text, tool_call_id=call_id)
                )

    structured = model.with_structured_output(schema, include_raw=True)
    final = structured.invoke(
        messages
        + [
            HumanMessage(
                content="Using ALL the evidence gathered above (tools + memory), "
                "produce the final structured result now."
            )
        ]
    )
    parsed = final.get("parsed") or schema()
    token_records.append(_usage_record(step, tier, final.get("raw"), model_override))
    return parsed, tool_records, token_records


def invoke_vision(
    step: str,
    system: str,
    user_text: str,
    image_data_urls: list[str],
    tier: str = "vision",
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """Send page images to a vision-capable OpenRouter model for OCR."""
    model = get_model(tier, temperature)
    content: list[dict] = [{"type": "text", "text": user_text}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    message = model.invoke(
        [SystemMessage(content=system), HumanMessage(content=content)]
    )
    return (message.content or ""), _usage_record(step, tier, message)

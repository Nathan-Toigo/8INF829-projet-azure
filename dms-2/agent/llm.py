"""OpenAI (ChatOpenAI) wrapper with per-call token usage capture.

Every call returns a token-usage record that nodes append to the ``AgentState``
token ledger, mirroring the token-accounting style of ``dms/llm.py``. Using
``ChatOpenAI`` (rather than the raw OpenAI client) means calls are natively traced
by LangSmith and tools can be bound directly.
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

import config
from agent.mcp_client import tool_result_to_text

_models: dict[float, ChatOpenAI] = {}


def get_model(temperature: float = 0.2) -> ChatOpenAI:
    if temperature not in _models:
        _models[temperature] = ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            temperature=temperature,
            api_key=config.require_openai_key(),
        )
    return _models[temperature]


def _usage_record(step: str, message: Any) -> dict:
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "step": step,
        "model": config.OPENAI_CHAT_MODEL,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


async def ainvoke_text(
    step: str, system: str, user: str, temperature: float = 0.3
) -> tuple[str, dict]:
    model = get_model(temperature)
    message = await model.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return (message.content or ""), _usage_record(step, message)


async def ainvoke_structured(
    step: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    temperature: float = 0.2,
) -> tuple[BaseModel, dict]:
    model = get_model(temperature)
    structured = model.with_structured_output(schema, include_raw=True)
    result = await structured.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    parsed = result.get("parsed")
    if parsed is None:  # fall back to an empty instance on parse failure
        parsed = schema()
    return parsed, _usage_record(step, result.get("raw"))


async def run_agentic_step(
    step: str,
    system: str,
    user: str,
    tools: list,
    schema: type[BaseModel],
    max_iters: int = 4,
    temperature: float = 0.2,
) -> tuple[BaseModel, list[dict], list[dict]]:
    """ReAct-style step: bind the MCP tools to ChatOpenAI and let the model
    decide which tools to call (and with what arguments) to dig further and
    enrich the memory-seeded context, then emit a structured result.

    Returns (parsed_schema, tool_call_records, token_records). Token usage from
    every LLM turn (exploration + final structuring) is captured.
    """
    model = get_model(temperature)
    by_name = {t.name: t for t in tools}
    messages: list[BaseMessage] = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    tool_records: list[dict] = []
    token_records: list[dict] = []

    # --- Exploration phase: the model may call MCP tools iteratively. ---
    if tools:
        llm_with_tools = model.bind_tools(tools)
        for _ in range(max_iters):
            ai = await llm_with_tools.ainvoke(messages)
            token_records.append(_usage_record(step, ai))
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
                        result = await tool.ainvoke(args)
                        result_text = tool_result_to_text(result)
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

    # --- Structuring phase: force a structured result from gathered evidence. ---
    structured = model.with_structured_output(schema, include_raw=True)
    final = await structured.ainvoke(
        messages
        + [
            HumanMessage(
                content="Using ALL the evidence gathered above (tools + memory), "
                "produce the final structured result now."
            )
        ]
    )
    parsed = final.get("parsed") or schema()
    token_records.append(_usage_record(step, final.get("raw")))
    return parsed, tool_records, token_records


async def run_agentic_text(
    step: str,
    system: str,
    user: str,
    tools: list,
    max_iters: int = 4,
    temperature: float = 0.3,
) -> tuple[str, list[dict], list[dict]]:
    """ReAct-style step returning free text (used for follow-up Q&A)."""
    model = get_model(temperature)
    by_name = {t.name: t for t in tools}
    messages: list[BaseMessage] = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    tool_records: list[dict] = []
    token_records: list[dict] = []
    final_text = ""

    if not tools:
        ai = await model.ainvoke(messages)
        token_records.append(_usage_record(step, ai))
        return (ai.content or ""), tool_records, token_records

    llm_with_tools = model.bind_tools(tools)
    for _ in range(max_iters):
        ai = await llm_with_tools.ainvoke(messages)
        token_records.append(_usage_record(step, ai))
        messages.append(ai)
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            final_text = ai.content or ""
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
                    result = await tool.ainvoke(args)
                    result_text = tool_result_to_text(result)
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
            messages.append(ToolMessage(content=result_text, tool_call_id=call_id))

    if not final_text:
        ai = await model.ainvoke(
            messages + [HumanMessage(content="Provide your final answer now.")]
        )
        token_records.append(_usage_record(step, ai))
        final_text = ai.content or ""
    return final_text, tool_records, token_records

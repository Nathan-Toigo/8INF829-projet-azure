"""Simple conversational agent: Ollama loop with MCP tool calls."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

import config
from agent.mcp_client import call_tool_async
from ollama_client import chat_completion_detailed

SYSTEM_PROMPT = """You are a helpful clinical-document assistant.
You have access to tools including RAG search over indexed medical records.

Guidelines:
- For factual questions about patient records, call rag_ask (or rag_search then summarize).
- Always pass chunk_method="{chunk_method}" to rag_ask, rag_search, and index_stats.
- top_k must be an integer (e.g. 5), never a quoted string.
- NEVER describe tool calls in plain text and NEVER output JSON tool plans in your reply.
  Use the native tool-calling API only.
- After tool results are returned, answer the user directly with facts and source file names.
- Use calculate for simple arithmetic (e.g. '(30+1)*19').
- If the index is empty, suggest running ingest_documents.
- Be concise and respond in the same language as the user."""

NUDGE_RETRY = (
    "Stop describing tools or outputting JSON call examples. "
    "Either invoke rag_ask via tool calling now, or give the user a direct final "
    "answer using the tool results already in this conversation."
)

_PSEUDO_ANSWER_MARKERS = (
    '"name": "rag_ask"',
    '"name":"rag_ask"',
    '"parameters":',
    "let me try again",
    "correct parameters",
    "top_k parameter",
    "requires a specific format",
    "i will call",
    "i'll call the",
    "tool requires",
)

# Ollama tool schemas aligned with mcp_server/server.py
OLLAMA_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "rag_ask",
            "description": "Answer a question using the full RAG pipeline (retrieve + generate).",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user question"},
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve",
                        "default": 5,
                    },
                    "chunk_method": {
                        "type": "string",
                        "description": "Chunking strategy: fixed_chars, paragraph, page, words_250",
                        "default": "fixed_chars",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Semantic search over indexed documents without generating an answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "chunk_method": {
                        "type": "string",
                        "default": "fixed_chars",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_documents",
            "description": "Index PDF/DOCX files from docs/ into Chroma.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clear": {
                        "type": "boolean",
                        "description": "Clear existing index before ingest",
                        "default": True,
                    },
                    "chunk_method": {
                        "type": "string",
                        "default": "fixed_chars",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "index_stats",
            "description": "Return indexed chunk count and source files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_method": {
                        "type": "string",
                        "default": "fixed_chars",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression, e.g. '(30+1)*19'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression using digits, +, -, *, /, %, ** and parentheses",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the current UTC time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_words",
            "description": "Count words in a text string.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "echo_message",
            "description": "Echo a message back (demo tool).",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_document_sources",
            "description": "List PDF/DOCX files available in docs/.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _ollama_chat(messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.OLLAMA_CHAT_MODEL,
        "messages": messages,
        "tools": OLLAMA_TOOLS,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    with httpx.Client(
        base_url=config.OLLAMA_BASE_URL.rstrip("/"),
        timeout=httpx.Timeout(config.OLLAMA_CHAT_TIMEOUT, connect=60.0),
    ) as client:
        r = client.post("/api/chat", json=payload)
        if r.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{config.OLLAMA_CHAT_MODEL}' not found. "
                f"Pull it: docker compose exec ollama ollama pull {config.OLLAMA_CHAT_MODEL}"
            ) from None
        r.raise_for_status()
        return r.json()


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _normalize_tool_args(
    name: str,
    args: dict[str, Any],
    *,
    chunk_method: str,
) -> dict[str, Any]:
    """Coerce types and fill defaults before MCP invocation."""
    out = dict(args)
    if "parameters" in out and isinstance(out["parameters"], dict):
        out = {**out, **out.pop("parameters")}
    if "arguments" in out and isinstance(out["arguments"], dict):
        out = {**out, **out.pop("arguments")}

    if name in ("rag_ask", "rag_search", "index_stats", "ingest_documents"):
        out.setdefault("chunk_method", chunk_method)

    if name == "rag_ask":
        if "question" not in out and "query" in out:
            out["question"] = out.pop("query")
        if "top_k" in out:
            try:
                out["top_k"] = max(1, int(out["top_k"]))
            except (TypeError, ValueError):
                out["top_k"] = config.TOP_K

    if name == "rag_search" and "top_k" in out:
        try:
            out["top_k"] = max(1, int(out["top_k"]))
        except (TypeError, ValueError):
            out["top_k"] = config.TOP_K

    return out


def _looks_like_pseudo_tool_answer(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if any(marker in lower for marker in _PSEUDO_ANSWER_MARKERS):
        return True
    if stripped.startswith("{") and re.search(r'"name"\s*:', stripped):
        return True
    return False


def _has_rag_tool_result(tool_calls_log: list[dict[str, Any]]) -> bool:
    for tc in tool_calls_log:
        if tc.get("tool") not in ("rag_ask", "rag_search"):
            continue
        preview = tc.get("result_preview") or ""
        if tc["tool"] == "rag_ask" and '"answer"' in preview:
            return True
        if tc["tool"] == "rag_search" and preview not in ("[]", ""):
            return True
    return False


def _is_valid_final_answer(
    answer: str,
    tool_calls_log: list[dict[str, Any]],
) -> bool:
    text = (answer or "").strip()
    if len(text) < 20:
        return False
    if _looks_like_pseudo_tool_answer(text):
        return False
    if _has_rag_tool_result(tool_calls_log):
        return True
    if tool_calls_log and not _looks_like_pseudo_tool_answer(text):
        return True
    lower = text.lower()
    if any(w in lower for w in ("rag_ask", "top_k", "tool call", "parameters")):
        return False
    return len(text) >= 40


def _force_final_answer(messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    result = chat_completion_detailed(
        messages
        + [
            {
                "role": "user",
                "content": (
                    "Using ONLY the tool results above, write a direct answer for the user. "
                    "Cite source files when available. Do not mention tools or JSON."
                ),
            }
        ]
    )
    usage = {
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "total_tokens": result.get("total_tokens"),
    }
    return result["content"], usage


class ChatAgent:
    """Minimal ReAct agent: Ollama decides which MCP tools to call."""

    def __init__(
        self,
        history: list[dict[str, str]] | None = None,
        chunk_method: str = "fixed_chars",
    ):
        self.history = list(history or [])
        self.chunk_method = chunk_method

    async def chat(self, user_message: str) -> dict[str, Any]:
        system = SYSTEM_PROMPT.format(chunk_method=self.chunk_method)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            *self.history,
            {"role": "user", "content": user_message},
        ]
        tool_calls_log: list[dict[str, Any]] = []
        token_records: list[dict[str, Any]] = []
        final_answer = ""
        max_rounds = config.AGENT_MAX_TOOL_ITERATIONS

        for round_idx in range(max_rounds):
            data = _ollama_chat(messages)
            msg = data.get("message") or {}
            usage = {
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
                "round": round_idx + 1,
            }
            if usage["prompt_tokens"] is not None and usage["completion_tokens"] is not None:
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            token_records.append(usage)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                candidate = (msg.get("content") or "").strip()
                if _is_valid_final_answer(candidate, tool_calls_log):
                    final_answer = candidate
                    messages.append({"role": "assistant", "content": final_answer})
                    break

                messages.append({"role": "assistant", "content": candidate})
                messages.append({"role": "user", "content": NUDGE_RETRY})
                continue

            messages.append(msg)
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                args = _normalize_tool_args(
                    name,
                    _parse_tool_args(fn.get("arguments")),
                    chunk_method=self.chunk_method,
                )
                t0 = time.perf_counter()
                try:
                    tool_result = await call_tool_async(name, args)
                    if isinstance(tool_result, (dict, list)):
                        result_text = json.dumps(tool_result, ensure_ascii=False)
                    else:
                        result_text = str(tool_result)
                except Exception as exc:
                    result_text = f"[tool '{name}' error: {exc}]"
                latency_ms = (time.perf_counter() - t0) * 1000
                tool_calls_log.append(
                    {
                        "tool": name,
                        "args": args,
                        "latency_ms": round(latency_ms, 2),
                        "result_preview": result_text[:600],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": result_text,
                        "name": name,
                    }
                )
        else:
            final_answer, usage = _force_final_answer(messages)
            usage["round"] = max_rounds + 1
            token_records.append(usage)
            messages.append({"role": "assistant", "content": final_answer})

        if not final_answer:
            final_answer, usage = _force_final_answer(messages)
            usage["round"] = "fallback"
            token_records.append(usage)

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": final_answer})

        return {
            "answer": final_answer,
            "tool_calls": tool_calls_log,
            "token_records": token_records,
            "rounds_used": len(token_records),
        }

    def reset(self) -> None:
        self.history.clear()

    def set_chunk_method(self, chunk_method: str) -> None:
        self.chunk_method = chunk_method

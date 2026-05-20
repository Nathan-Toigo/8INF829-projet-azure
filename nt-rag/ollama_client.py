"""Ollama HTTP API client (embeddings + chat)."""

from __future__ import annotations

import httpx

import config


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=config.OLLAMA_BASE_URL.rstrip("/"),
        timeout=config.OLLAMA_TIMEOUT,
    )


def check_ollama() -> None:
    """Verify Ollama is reachable and required models are present."""
    try:
        with _client() as client:
            r = client.get("/api/tags")
            r.raise_for_status()
            names = {m["name"].split(":")[0] for m in r.json().get("models", [])}
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. "
            "Start Docker: cd rag && docker compose up -d"
        ) from e

    missing = []
    for model in (config.OLLAMA_EMBED_MODEL, config.OLLAMA_CHAT_MODEL):
        base = model.split(":")[0]
        if base not in names and model not in names:
            missing.append(model)
    if missing:
        raise RuntimeError(
            f"Missing Ollama models: {', '.join(missing)}. "
            f"Run: docker compose exec ollama ollama pull {missing[0]}"
        )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    with _client() as client:
        r = client.post(
            "/api/embed",
            json={"model": config.OLLAMA_EMBED_MODEL, "input": texts},
        )
        r.raise_for_status()
        data = r.json()
    return data["embeddings"]


def chat_completion(messages: list[dict[str, str]]) -> str:
    with _client() as client:
        r = client.post(
            "/api/chat",
            json={
                "model": config.OLLAMA_CHAT_MODEL,
                "messages": messages,
                "stream": False,
            },
        )
        r.raise_for_status()
        data = r.json()
    return data.get("message", {}).get("content", "") or ""

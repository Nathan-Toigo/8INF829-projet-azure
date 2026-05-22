"""Ollama HTTP API client (embeddings + chat)."""

from __future__ import annotations

import httpx

import config


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=config.OLLAMA_BASE_URL.rstrip("/"),
        timeout=config.OLLAMA_TIMEOUT,
    )


def check_ollama(
    *,
    embed_model: str | None = None,
    chat_model: str | None = None,
) -> None:
    """Verify Ollama is reachable and required models are present."""
    embed = embed_model or config.OLLAMA_EMBED_MODEL
    chat = chat_model or config.OLLAMA_CHAT_MODEL
    try:
        with _client() as client:
            r = client.get("/api/tags")
            r.raise_for_status()
            names = {m["name"].split(":")[0] for m in r.json().get("models", [])}
            full_names = {m["name"] for m in r.json().get("models", [])}
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. "
            "Start Docker: cd nt-rag && docker compose up -d"
        ) from e

    missing = []
    for model in (embed, chat):
        base = model.split(":")[0]
        if base not in names and model not in full_names:
            missing.append(model)
    if missing:
        raise RuntimeError(
            f"Missing Ollama models: {', '.join(missing)}. "
            f"Run: docker compose exec ollama ollama pull {missing[0]}"
        )


def embed_texts(
    texts: list[str],
    *,
    model: str | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    embed_model = model or config.OLLAMA_EMBED_MODEL
    with _client() as client:
        r = client.post(
            "/api/embed",
            json={"model": embed_model, "input": texts},
        )
        r.raise_for_status()
        data = r.json()
    return data["embeddings"]


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    chat_model = model or config.OLLAMA_CHAT_MODEL
    with _client() as client:
        r = client.post(
            "/api/chat",
            json={
                "model": chat_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        r.raise_for_status()
        data = r.json()
    return data.get("message", {}).get("content", "") or ""

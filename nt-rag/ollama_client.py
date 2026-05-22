"""Ollama HTTP API client (embeddings + chat)."""

from __future__ import annotations

import time

import httpx

import config

# Transient errors when Ollama closes the socket (OOM, reload, long generation)
_TRANSIENT_CHAT_ERRORS = (
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
)


def _http_timeout(seconds: float) -> httpx.Timeout:
    """Separate connect vs read so long generations do not abort too early."""
    return httpx.Timeout(
        timeout=seconds,
        connect=60.0,
        read=seconds,
        write=60.0,
        pool=30.0,
    )


def _client(timeout: float | None = None) -> httpx.Client:
    sec = timeout if timeout is not None else config.OLLAMA_TIMEOUT
    return httpx.Client(
        base_url=config.OLLAMA_BASE_URL.rstrip("/"),
        timeout=_http_timeout(sec),
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
        with _client(timeout=30) as client:
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


def truncate_for_embed(text: str, max_chars: int | None = None) -> str:
    """Keep embed inputs under model context (avoids Ollama 400 on long page chunks)."""
    limit = max_chars if max_chars is not None else config.EMBED_MAX_CHARS
    if len(text) <= limit:
        return text
    return text[:limit]


def embed_texts(
    texts: list[str],
    *,
    model: str | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    embed_model = model or config.OLLAMA_EMBED_MODEL
    prepared = [truncate_for_embed(t) for t in texts]
    payload = {"model": embed_model, "input": prepared, "truncate": True}
    with _client() as client:
        r = client.post("/api/embed", json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama embed HTTP {e.response.status_code}: "
                f"{e.response.text[:500]}. "
                f"If chunks are very long, lower EMBED_MAX_CHARS (now {config.EMBED_MAX_CHARS}) "
                "or avoid chunk_method page without splitting."
            ) from e
        data = r.json()
    return data["embeddings"]


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    timeout: float | None = None,
    max_retries: int | None = None,
    num_ctx: int | None = None,
) -> str:
    chat_model = model or config.OLLAMA_CHAT_MODEL
    read_timeout = timeout if timeout is not None else config.OLLAMA_CHAT_TIMEOUT
    retries = config.OLLAMA_CHAT_RETRIES if max_retries is None else max_retries

    payload: dict = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if num_ctx is not None:
        payload["options"]["num_ctx"] = num_ctx

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with _client(timeout=read_timeout) as client:
                r = client.post("/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
            return data.get("message", {}).get("content", "") or ""
        except _TRANSIENT_CHAT_ERRORS as e:
            last_error = e
            if attempt < retries:
                wait = min(2**attempt * 2, 15)
                print(
                    f"  Ollama chat retry {attempt + 1}/{retries} "
                    f"after {type(e).__name__} (wait {wait}s)...",
                    flush=True,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(
                "Ollama closed the connection during chat. Common causes: "
                "prompt too large (lower MAX_FULL_DOC_CHARS), GPU/RAM OOM, or "
                "container restart. Check: docker logs rag_ollama"
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama HTTP {e.response.status_code}: {e.response.text[:500]}"
            ) from e

    raise RuntimeError("Ollama chat failed") from last_error

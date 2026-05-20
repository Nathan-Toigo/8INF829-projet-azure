"""PostgreSQL + pgvector persistence and search."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from config import DATABASE_URL, CHUNK_METHODS

VECTOR_DIM = 1536


def _ensure_vector_extension(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()


@contextmanager
def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    _ensure_vector_extension(conn)
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    chunk_method TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    page_number INT,
                    word_count INT,
                    content TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding vector({VECTOR_DIM}) NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (source_file, chunk_method, chunk_index, embedding_model)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_method
                ON document_chunks (chunk_method)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops)
                """
            )


def clear_chunks() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE document_chunks RESTART IDENTITY")


def insert_chunks(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO document_chunks (
            source_file, chunk_method, chunk_index, page_number,
            word_count, content, embedding_model, embedding, metadata
        ) VALUES %s
        ON CONFLICT (source_file, chunk_method, chunk_index, embedding_model)
        DO UPDATE SET
            content = EXCLUDED.content,
            word_count = EXCLUDED.word_count,
            page_number = EXCLUDED.page_number,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata
    """
    values = [
        (
            r["source_file"],
            r["chunk_method"],
            r["chunk_index"],
            r.get("page_number"),
            r.get("word_count"),
            r["content"],
            r["embedding_model"],
            r["embedding"],
            json.dumps(r.get("metadata", {})),
        )
        for r in rows
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, values, template="(%s,%s,%s,%s,%s,%s,%s,%s,%s)")
    return len(rows)


def search_similar(
    query_embedding: list[float],
    embedding_model: str,
    chunk_method: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Cosine distance search; returns similarity = 1 - distance."""
    filters = ["embedding_model = %s"]
    if chunk_method:
        filters.append("chunk_method = %s")
    where = " AND ".join(filters)
    sql = f"""
        SELECT
            id, source_file, chunk_method, chunk_index, page_number,
            word_count, content, embedding_model,
            (embedding <=> %s::vector) AS distance,
            1 - (embedding <=> %s::vector) AS similarity
        FROM document_chunks
        WHERE {where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params: list[Any] = [query_embedding, query_embedding, embedding_model]
    if chunk_method:
        params.append(chunk_method)
    params.extend([query_embedding, limit])

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def count_chunks() -> dict[str, int]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_method, embedding_model, COUNT(*)
                FROM document_chunks
                GROUP BY chunk_method, embedding_model
                ORDER BY chunk_method, embedding_model
                """
            )
            return {
                f"{method}|{model}": count
                for method, model, count in cur.fetchall()
            }


def wait_for_db(max_attempts: int = 30, delay_sec: float = 2.0) -> bool:
    import time

    for _ in range(max_attempts):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.close()
            return True
        except psycopg2.OperationalError:
            time.sleep(delay_sec)
    return False

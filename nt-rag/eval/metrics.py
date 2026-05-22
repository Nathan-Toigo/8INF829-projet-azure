"""Timing helpers and aggregation for benchmark results."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    bucket: dict[str, float] = {}
    t0 = time.perf_counter()
    yield bucket
    bucket["elapsed_sec"] = round(time.perf_counter() - t0, 3)


def aggregate_experiment_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average numeric metrics across questions for one experiment."""
    if not rows:
        return {}

    def avg(key: str) -> float | None:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None

    score_keys = [
        "accuracy_score",
        "quality_score",
        "thoroughness_score",
        "global_accuracy_score",
        "golden_match_score",
    ]
    out: dict[str, Any] = {
        "question_count": len(rows),
        "avg_query_embed_ms": avg("query_embed_ms"),
        "avg_retrieve_ms": avg("retrieve_ms"),
        "avg_chat_ms": avg("chat_ms"),
        "avg_total_question_sec": avg("total_question_sec"),
        "avg_full_chart_sec": avg("full_chart_sec"),
        "avg_judge_sec": avg("judge_sec"),
    }
    for sk in score_keys:
        out[f"avg_{sk}"] = avg(sk)
    return out

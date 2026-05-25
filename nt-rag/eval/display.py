"""Console progress and metrics for benchmark runs."""

from __future__ import annotations

from typing import Any


def log_banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}", flush=True)


def log_step(label: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"  >> {label}{suffix}", flush=True)


def log_done(label: str, elapsed_sec: float | None = None, extra: str = "") -> None:
    parts = [f"  OK {label}"]
    if elapsed_sec is not None:
        parts.append(f"({elapsed_sec:.2f}s)")
    if extra:
        parts.append(f"— {extra}")
    print(" ".join(parts), flush=True)


def log_warn(msg: str) -> None:
    print(f"  WARN {msg}", flush=True)


def log_metrics_block(title: str, metrics: dict[str, Any]) -> None:
    print(f"  [{title}]", flush=True)
    for key, val in metrics.items():
        if val is None:
            continue
        print(f"      {key}: {val}", flush=True)


def log_retrieval(
    retrieved: list[dict[str, Any]],
    *,
    top_k: int,
) -> None:
    if not retrieved:
        print("      (no chunks retrieved)", flush=True)
        return
    print(f"      top_{top_k} chunks (cosine distance, lower = closer):", flush=True)
    for i, ch in enumerate(retrieved, start=1):
        dist = ch.get("distance")
        dist_s = f"{dist:.4f}" if isinstance(dist, (int, float)) else "n/a"
        src = ch.get("source_file", "?")
        page = ch.get("page_index", "?")
        preview = (ch.get("preview") or "").replace("\n", " ")[:80]
        print(f"        #{i} d={dist_s}  {src} p.{page}  {preview}", flush=True)


def log_judge_scores(scores: dict[str, Any], *, prefix: str = "judge") -> None:
    if not scores:
        log_warn(f"{prefix}: no scores")
        return
    if scores.get("parse_error"):
        log_warn(f"{prefix}: JSON parse failed")
        raw = str(scores.get("parse_error", ""))[:120]
        print(f"      raw: {raw}...", flush=True)
        return
    if scores.get("_parse_partial"):
        print(f"      ({prefix}: scores recovered from partial JSON)", flush=True)
    for key in (
        "accuracy_score",
        "quality_score",
        "thoroughness_score",
        "global_accuracy_score",
        "golden_match_score",
    ):
        if key in scores and scores[key] is not None:
            print(f"      {key}: {scores[key]}", flush=True)


def log_question_summary(row: dict[str, Any]) -> None:
    parts = [
        f"chat={row.get('chat_ms')}ms",
        f"tokens={row.get('total_tokens')}",
        f"top1={row.get('top1_distance')}",
    ]
    print(f"  --- summary: {', '.join(parts)}", flush=True)


def log_experiment_summary(summary: dict[str, Any]) -> None:
    print(f"\n  Experiment summary ({summary.get('experiment_id')}):", flush=True)
    for key in (
        "question_count",
        "avg_total_question_sec",
        "avg_query_embed_ms",
        "avg_retrieve_ms",
        "avg_chat_ms",
        "avg_top1_distance",
        "avg_retrieve_distance",
        "avg_full_chart_sec",
        "avg_judge_sec",
        "avg_accuracy_score",
        "avg_quality_score",
        "avg_thoroughness_score",
        "avg_global_accuracy_score",
        "avg_golden_match_score",
    ):
        val = summary.get(key)
        if val is not None:
            print(f"      {key}: {val}", flush=True)

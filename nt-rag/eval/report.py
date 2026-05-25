"""Write benchmark CSV and Markdown reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

SUMMARY_ROW_COLUMNS = [
    "chunk_method",
    "question",
    "chat_ms",
    "rag_answer",
    "total_tokens",
    "top1_distance",
]


def detail_row_to_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_method": row.get("chunk_method"),
        "question": row.get("question"),
        "chat_ms": row.get("chat_ms"),
        "rag_answer": row.get("rag_answer"),
        "total_tokens": row.get("total_tokens"),
        "top1_distance": row.get("top1_distance"),
    }


def write_summary_csv(path: Path, detail_rows: list[dict[str, Any]]) -> None:
    if not detail_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [detail_row_to_summary(r) for r in detail_rows]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_ROW_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _truncate(text: str, max_len: int = 80) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def write_report_md(
    path: Path,
    detail_rows: list[dict[str, Any]],
    hardware: dict[str, Any],
) -> None:
    lines = [
        "# RAG Benchmark Report",
        "",
        "## Hardware",
        "",
        f"- Host profile: {hardware.get('host_profile')}",
        f"- Platform: {hardware.get('platform')}",
        f"- CPU cores: {hardware.get('cpu_count')}",
        f"- RAM (GB): {hardware.get('ram_total_gb')} total / {hardware.get('ram_available_gb')} available",
        f"- GPU: {hardware.get('gpu_nvidia_smi') or 'N/A'}",
        "",
        "## Answers",
        "",
        "| Chunk | Question | Chat (ms) | Tokens | Top1 dist | Answer (preview) |",
        "|-------|----------|-----------|--------|-----------|------------------|",
    ]
    for r in detail_rows:
        lines.append(
            f"| {r.get('chunk_method', '')} | {_truncate(r.get('question', ''), 40)} | "
            f"{r.get('chat_ms', '')} | {r.get('total_tokens', '')} | "
            f"{r.get('top1_distance', '')} | {_truncate(r.get('rag_answer', ''), 60)} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

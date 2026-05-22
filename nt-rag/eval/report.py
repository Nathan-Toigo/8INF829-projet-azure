"""Write benchmark CSV and Markdown reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for s in summaries:
        for k in s:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries)


def write_report_md(
    path: Path,
    summaries: list[dict[str, Any]],
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
        f"- Ollama GPU mode: {hardware.get('ollama_gpu_mode')}",
        "",
        "## Experiment summaries",
        "",
        "| Experiment | Chunk | Avg RAG (s) | Avg top1 dist | Avg accuracy | Avg golden |",
        "|------------|-------|-------------|---------------|--------------|------------|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.get('experiment_id', '')} | {s.get('chunk_method', '')} | "
            f"{s.get('avg_total_question_sec', '')} | {s.get('avg_top1_distance', '')} | "
            f"{s.get('avg_global_accuracy_score', '')} | "
            f"{s.get('avg_golden_match_score', '')} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

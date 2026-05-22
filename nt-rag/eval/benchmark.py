#!/usr/bin/env python3
"""Run RAG benchmark from YAML experiment matrix."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import config
from eval.full_chart import answer_with_full_documents
from eval.hardware import capture_hardware_snapshot
from eval.judge import judge_rag_vs_full, judge_vs_golden
from eval.metrics import aggregate_experiment_rows
from eval.questions import load_questions
from eval.report import write_jsonl, write_report_md, write_summary_csv
from ingest import run_ingest
from ollama_client import check_ollama
from query import ask_with_metrics

RAG_ROOT = Path(__file__).resolve().parent.parent


def load_benchmark_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_golden(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_question_row(
    experiment: dict[str, Any],
    question: dict[str, Any],
    rag: dict[str, Any],
    full: dict[str, Any] | None,
    judge_full: dict[str, Any] | None,
    judge_golden: dict[str, Any] | None,
    ingest_metrics: dict[str, Any],
) -> dict[str, Any]:
    m = rag.get("metrics", {})
    row: dict[str, Any] = {
        "experiment_id": experiment["id"],
        "question_id": question["id"],
        "question": question["question"],
        "chunk_method": experiment.get("chunk_method", "fixed_chars"),
        "chat_model": experiment.get("chat_model", config.OLLAMA_CHAT_MODEL),
        "embed_model": experiment.get("embed_model", config.OLLAMA_EMBED_MODEL),
        "top_k": experiment.get("top_k", config.TOP_K),
        "collection_name": ingest_metrics.get("collection_name"),
        "rag_answer": rag.get("answer"),
        "query_embed_ms": m.get("query_embed_ms"),
        "retrieve_ms": m.get("retrieve_ms"),
        "chat_ms": m.get("chat_ms"),
        "total_question_sec": round(m.get("total_ms", 0) / 1000, 3),
        "ingest_total_sec": ingest_metrics.get("ingest_total_sec"),
    }
    if full:
        row["full_chart_answer"] = full.get("answer")
        row["full_chart_sec"] = full.get("metrics", {}).get("full_chart_sec")
    if judge_full:
        scores = judge_full.get("scores", {})
        row["accuracy_score"] = scores.get("accuracy_score")
        row["quality_score"] = scores.get("quality_score")
        row["thoroughness_score"] = scores.get("thoroughness_score")
        row["global_accuracy_score"] = scores.get("global_accuracy_score")
        row["judge_sec"] = judge_full.get("metrics", {}).get("judge_sec")
    if judge_golden:
        gs = judge_golden.get("scores", {})
        row["golden_match_score"] = gs.get("golden_match_score")
        row["has_golden"] = True
    else:
        row["has_golden"] = False
    return row


def run_benchmark(
    config_path: Path,
    *,
    only_experiment: str | None = None,
    dry_run: bool = False,
    all_questions: bool = False,
    skip_ingest: bool = False,
) -> Path:
    bench = load_benchmark_config(config_path)
    host_profile = bench.get("host_profile", "default")
    run_full = bench.get("run_full_chart", True)
    run_judge = bench.get("run_judge", True)

    questions_path = RAG_ROOT / bench.get("questions", "eval/questions.json")
    golden_path = RAG_ROOT / bench.get("golden_answers", "eval/golden_answers.json")
    questions = load_questions(questions_path, all_questions=all_questions)
    if dry_run:
        questions = questions[:1]

    golden = load_golden(golden_path)
    experiments = bench.get("experiments", [])
    if only_experiment:
        experiments = [e for e in experiments if e.get("id") == only_experiment]
        if not experiments:
            raise ValueError(f"Experiment not found: {only_experiment}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = config.RESULTS_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    hw = capture_hardware_snapshot(host_profile=host_profile)
    (out_dir / "hardware.json").write_text(
        json.dumps(hw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "config_resolved.yaml").write_text(
        yaml.dump(bench, allow_unicode=True), encoding="utf-8"
    )

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for exp in experiments:
        exp_id = exp["id"]
        chunk_method = exp.get("chunk_method", "fixed_chars")
        embed_model = exp.get("embed_model", config.OLLAMA_EMBED_MODEL)
        chat_model = exp.get("chat_model", config.OLLAMA_CHAT_MODEL)
        coll = config.collection_name_for(chunk_method, embed_model)

        print(f"\n=== Experiment: {exp_id} ===")
        check_ollama(embed_model=embed_model, chat_model=chat_model)

        if skip_ingest:
            ingest_metrics = {
                "collection_name": coll,
                "chunk_method": chunk_method,
                "embed_model": embed_model,
            }
        else:
            ingest_metrics = run_ingest(
                clear=True,
                chunk_method=chunk_method,
                chunk_size=exp.get("chunk_size"),
                chunk_overlap=exp.get("chunk_overlap"),
                embed_model=embed_model,
                collection_name=coll,
            )

        exp_rows: list[dict[str, Any]] = []
        for q in questions:
            qid = q["id"]
            qtext = q["question"]
            print(f"  Question {qid}...")

            rag = ask_with_metrics(
                qtext,
                top_k=exp.get("top_k", config.TOP_K),
                collection_name=coll,
                embed_model=embed_model,
                chat_model=chat_model,
            )

            full_result = None
            judge_full_result = None
            judge_golden_result = None

            if run_full:
                full_result = answer_with_full_documents(qtext, chat_model=chat_model)

            if run_judge and full_result:
                judge_full_result = judge_rag_vs_full(
                    qtext,
                    rag["answer"],
                    full_result["answer"],
                    rag["rag_context"],
                    judge_model=exp.get("judge_model"),
                )

            if run_judge and qid in golden:
                judge_golden_result = judge_vs_golden(
                    qtext,
                    rag["answer"],
                    golden[qid]["answer"],
                    rag["rag_context"],
                    judge_model=exp.get("judge_model"),
                )

            row = _flatten_question_row(
                exp, q, rag, full_result, judge_full_result, judge_golden_result, ingest_metrics
            )
            exp_rows.append(row)
            all_rows.append(row)

        summary = aggregate_experiment_rows(exp_rows)
        summary["experiment_id"] = exp_id
        summary["chunk_method"] = chunk_method
        summary["chat_model"] = chat_model
        summary["embed_model"] = embed_model
        summary["host_profile"] = host_profile
        summaries.append(summary)

    write_jsonl(out_dir / "details.jsonl", all_rows)
    write_summary_csv(out_dir / "summary.csv", summaries)
    write_report_md(out_dir / "report.md", summaries, hw)
    print(f"\nResults written to {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="RAG benchmark runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=RAG_ROOT / "experiments" / "benchmark.yaml",
    )
    parser.add_argument("--only", dest="only_experiment", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(RAG_ROOT))
    run_benchmark(
        args.config,
        only_experiment=args.only_experiment,
        dry_run=args.dry_run,
        all_questions=args.all_questions,
        skip_ingest=args.skip_ingest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run RAG benchmark from YAML experiment matrix."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import config
from eval.display import (
    log_banner,
    log_done,
    log_question_summary,
    log_step,
)
from eval.hardware import capture_hardware_snapshot
from eval.questions import load_questions
from eval.report import write_report_md, write_summary_csv
from ingest import (
    ingest_metrics_for_existing_collection,
    run_ingest,
    should_skip_ingest,
)
from ollama_client import check_ollama
from query import ask_with_metrics
from store import collection_vector_count

RAG_ROOT = Path(__file__).resolve().parent.parent


def load_benchmark_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build_question_row(
    experiment: dict[str, Any],
    question: dict[str, Any],
    rag: dict[str, Any],
    ingest_metrics: dict[str, Any],
) -> dict[str, Any]:
    m = rag.get("metrics", {})
    retrieved = rag.get("retrieved_chunks") or []
    distances = [
        c["distance"]
        for c in retrieved
        if isinstance(c.get("distance"), (int, float))
    ]
    row: dict[str, Any] = {
        "experiment_id": experiment["id"],
        "question_id": question["id"],
        "chunk_method": experiment.get("chunk_method", "fixed_chars"),
        "question": question["question"],
        "chat_ms": m.get("chat_ms"),
        "rag_answer": rag.get("answer"),
        "prompt_tokens": m.get("prompt_tokens"),
        "completion_tokens": m.get("completion_tokens"),
        "total_tokens": m.get("total_tokens"),
        "top1_distance": round(distances[0], 4) if distances else None,
        "chat_model": experiment.get("chat_model", config.OLLAMA_CHAT_MODEL),
        "embed_model": experiment.get("embed_model", config.OLLAMA_EMBED_MODEL),
        "collection_name": ingest_metrics.get("collection_name"),
        "query_embed_ms": m.get("query_embed_ms"),
        "retrieve_ms": m.get("retrieve_ms"),
        "total_question_sec": round(m.get("total_ms", 0) / 1000, 3),
    }
    return row


def run_benchmark(
    config_path: Path,
    *,
    only_experiment: str | None = None,
    dry_run: bool = False,
    all_questions: bool = False,
    skip_ingest: bool = False,
    force_ingest: bool = False,
) -> Path:
    bench = load_benchmark_config(config_path)
    host_profile = bench.get("host_profile", "default")
    reuse_collection = bench.get("reuse_existing_collection", True)

    questions_path = RAG_ROOT / bench.get("questions", "eval/questions.json")
    questions = load_questions(questions_path, all_questions=all_questions)
    if dry_run:
        questions = questions[:1]

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

    log_banner("RAG benchmark")
    print(f"  Output: {out_dir}", flush=True)
    print(f"  Experiments: {len(experiments)}", flush=True)
    print(f"  Questions: {len(questions)} ({', '.join(q['id'] for q in questions)})", flush=True)
    print(
        f"  Flags: reuse_collection={reuse_collection} "
        f"skip_ingest={skip_ingest} force_ingest={force_ingest}",
        flush=True,
    )

    for exp in experiments:
        exp_id = exp["id"]
        chunk_method = exp.get("chunk_method", "fixed_chars")
        embed_model = exp.get("embed_model", config.OLLAMA_EMBED_MODEL)
        chat_model = exp.get("chat_model", config.OLLAMA_CHAT_MODEL)
        coll = config.collection_name_for(chunk_method, embed_model)

        log_banner(f"Experiment: {exp_id}")
        print(
            f"  chunk={chunk_method} embed={embed_model} chat={chat_model} "
            f"collection={coll}",
            flush=True,
        )
        log_step("check Ollama models")
        check_ollama(embed_model=embed_model, chat_model=chat_model)
        log_done("Ollama OK")

        n_existing = collection_vector_count(coll)
        do_ingest = True
        if skip_ingest:
            do_ingest = False
        elif force_ingest:
            do_ingest = True
        elif reuse_collection and should_skip_ingest(coll):
            do_ingest = False

        if not do_ingest:
            ingest_metrics = ingest_metrics_for_existing_collection(
                coll,
                chunk_method=chunk_method,
                embed_model=embed_model,
            )
            reason = "--skip-ingest" if skip_ingest else f"collection already has {n_existing} vectors"
            log_step("ingest", f"reused ({reason})")
            log_done("ingest", 0, f"{n_existing} vectors in {coll}")
        else:
            log_step("ingest", f"method={chunk_method}")
            ingest_metrics = run_ingest(
                clear=True,
                chunk_method=chunk_method,
                chunk_size=exp.get("chunk_size"),
                chunk_overlap=exp.get("chunk_overlap"),
                embed_model=embed_model,
                collection_name=coll,
            )
            log_done(
                "ingest",
                ingest_metrics.get("ingest_total_sec"),
                f"{ingest_metrics.get('chunks')} chunks -> {coll}",
            )

        for qi, q in enumerate(questions, start=1):
            qid = q["id"]
            qtext = q["question"]
            print(f"\n  --- Question {qi}/{len(questions)}: {qid} ---", flush=True)
            qpreview = qtext[:70] + ("..." if len(qtext) > 70 else "")
            print(f"      {qpreview}", flush=True)

            log_step("RAG answer")
            rag = ask_with_metrics(
                qtext,
                top_k=exp.get("top_k", config.TOP_K),
                collection_name=coll,
                embed_model=embed_model,
                chat_model=chat_model,
                verbose=True,
            )
            ans = rag.get("answer") or ""
            log_done(
                "RAG",
                rag.get("metrics", {}).get("total_ms", 0) / 1000,
                f"answer {len(ans)} chars",
            )

            row = _build_question_row(exp, q, rag, ingest_metrics)
            log_question_summary(row)
            all_rows.append(row)

    write_summary_csv(out_dir / "summary.csv", all_rows)
    write_report_md(out_dir / "report.md", all_rows, hw)
    log_banner("Benchmark complete")
    print(f"  Results: {out_dir}", flush=True)
    print(f"    summary.csv ({len(all_rows)} rows)", flush=True)
    print(f"    report.md", flush=True)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="RAG benchmark runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=RAG_ROOT / "experiments/benchmark.yaml",
    )
    parser.add_argument("--only", dest="only_experiment", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Never re-embed; reuse Chroma even if empty",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Always clear and re-embed collections",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(RAG_ROOT))
    run_benchmark(
        args.config,
        only_experiment=args.only_experiment,
        dry_run=args.dry_run,
        all_questions=args.all_questions,
        skip_ingest=args.skip_ingest,
        force_ingest=args.force_ingest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

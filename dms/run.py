#!/usr/bin/env python3
"""
End-to-end pipeline:
  1. Ingest RAG data (chunk + embed + store)
  2. Initialize LLM
  3. Per question: RAG answer, full-document answer, judge comparison
  4. Print answers, metrics, and final score matrix
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import time

import config
from database import count_chunks, wait_for_db
from ingest import run_ingest
from llm import evaluate_question, init_llm, print_rag_metrics_from_dict


def separator(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _score(scores: dict, key: str) -> str:
    val = scores.get(key)
    return str(val) if val is not None else "—"


def _tok(val: int | float | None) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.1f}"
    return str(val)


def print_score_matrix(qa_results: list[dict]) -> None:
    """Render final score matrix for all questions (scores + Full-Chart token usage)."""
    headers = [
        "Question ID",
        "Accuracy",
        "Quality",
        "Thoroughness",
        "Global",
        "FC Tokens",
        "FC +Δ vs RAG",
        "FC +Δ %",
        "FC % of Q",
    ]
    rows: list[list[str]] = []
    for item in qa_results:
        scores = item.get("judge", {}).get("scores", {})
        tu = item.get("token_usage", {})
        rows.append(
            [
                item["id"],
                _score(scores, "accuracy_score"),
                _score(scores, "quality_score"),
                _score(scores, "thoroughness_score"),
                _score(scores, "global_accuracy_score"),
                _tok(tu.get("full_chart_tokens")),
                _tok(tu.get("full_chart_additional_tokens")),
                _tok(tu.get("full_chart_additional_pct_vs_rag")),
                _tok(tu.get("full_chart_pct_of_question_tokens")),
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    separator("FINAL SCORE MATRIX — RAG vs FULL-CHART (Judge LLM)")
    print(
        "  FC Tokens = Full-Chart total | +Δ = additional vs RAG | "
        "+Δ % = (FC−RAG)/RAG×100 | FC % of Q = FC/(RAG+FC+Judge)×100"
    )
    print()
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))

    # Averages
    numeric_keys = [
        "accuracy_score",
        "quality_score",
        "thoroughness_score",
        "global_accuracy_score",
    ]
    avgs: dict[str, float] = {}
    for key in numeric_keys:
        vals = [
            item["judge"]["scores"][key]
            for item in qa_results
            if isinstance(item.get("judge", {}).get("scores", {}).get(key), (int, float))
        ]
        if vals:
            avgs[key] = sum(vals) / len(vals)

    if avgs:
        print("\nScore averages:")
        print(
            f"  Accuracy: {avgs.get('accuracy_score', 0):.1f}  "
            f"Quality: {avgs.get('quality_score', 0):.1f}  "
            f"Thoroughness: {avgs.get('thoroughness_score', 0):.1f}  "
            f"Global: {avgs.get('global_accuracy_score', 0):.1f}"
        )

    # Token usage totals / averages
    fc_tokens = [item["token_usage"]["full_chart_tokens"] for item in qa_results if item.get("token_usage")]
    rag_tokens = [item["token_usage"]["rag_tokens"] for item in qa_results if item.get("token_usage")]
    add_tokens = [
        item["token_usage"]["full_chart_additional_tokens"]
        for item in qa_results
        if item.get("token_usage")
    ]
    add_pcts = [
        item["token_usage"]["full_chart_additional_pct_vs_rag"]
        for item in qa_results
        if item.get("token_usage")
        and item["token_usage"]["full_chart_additional_pct_vs_rag"] is not None
    ]
    if fc_tokens:
        total_fc = sum(fc_tokens)
        total_rag = sum(rag_tokens)
        total_add = sum(add_tokens)
        print("\nToken usage totals (all questions):")
        print(f"  RAG tokens: {total_rag:,}")
        print(f"  Full-Chart tokens: {total_fc:,}")
        print(f"  Full-Chart additional vs RAG: {total_add:,} tokens")
        if total_rag:
            print(
                f"  Full-Chart additional vs RAG: "
                f"{(total_add / total_rag) * 100:.1f}%"
            )
        if add_pcts:
            print(f"  Mean Full-Chart +Δ % per question: {sum(add_pcts) / len(add_pcts):.1f}%")
        print("\n  FC = Full-Chart physician call | +Δ = extra tokens vs RAG | % of Q = share of question total (RAG+FC+Judge)")


def main() -> int:
    if not config.OPENAI_API_KEY:
        print(
            "ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        return 1

    if not wait_for_db():
        print(
            "ERROR: Cannot connect to PostgreSQL. Run ./init.sh to start Docker.",
            file=sys.stderr,
        )
        return 1

    pipeline_start = time.perf_counter()
    results: dict = {"steps": []}

    # Step 1: Ingest
    separator("STEP 1 — INGEST (chunk + embed + pgvector)")
    t0 = time.perf_counter()
    ingest_metrics = run_ingest(clear=True)
    ingest_elapsed = time.perf_counter() - t0
    ingest_metrics["elapsed_sec"] = round(ingest_elapsed, 2)
    ingest_metrics["db_counts"] = count_chunks()
    results["steps"].append({"ingest": ingest_metrics})
    print(f"\nIngest elapsed: {ingest_elapsed:.2f}s")
    print("DB chunk counts:", ingest_metrics["db_counts"])

    # Step 2: Initialize LLM
    separator("STEP 2 — INITIALIZE LLM")
    init_llm()
    print(f"  Chat model: {config.OPENAI_CHAT_MODEL}")
    print(f"  Embedding model: {config.OPENAI_EMBEDDING_MODEL}")
    print(f"  Available embedding engines for testing: {config.EMBEDDING_MODELS}")
    results["steps"].append(
        {
            "llm_init": {
                "chat_model": config.OPENAI_CHAT_MODEL,
                "embedding_model": config.OPENAI_EMBEDDING_MODEL,
                "embedding_models_available": config.EMBEDDING_MODELS,
            }
        }
    )

    # Step 3–5: Test questions (RAG, full docs, judge)
    separator("STEP 3–5 — PER QUESTION: RAG + FULL DOCS + JUDGE")
    qa_results = []

    for qid, question in config.TEST_QUESTIONS:
        separator(f"QUESTION [{qid}]")
        print(question)

        eval_result = evaluate_question(qid, question)
        qa_results.append(eval_result)

        rag_payload = eval_result["rag"]
        full_payload = eval_result["full_documents"]
        judge_payload = eval_result["judge"]

        # Step 3a: RAG
        print("\n  >>> STEP A — RAG-BASED ANSWER")
        print_rag_metrics_from_dict(rag_payload["rag"])
        print("\n  RAG LLM metrics:")
        for k, v in rag_payload["metrics"].items():
            print(f"    {k}: {v}")
        print("\n  --- RAG PHYSICIAN ANSWER ---")
        print(rag_payload["answer"])

        # Step 3b: Full documents
        print("\n  >>> STEP B — FULL-DOCUMENT ANSWER")
        meta = full_payload.get("corpus_meta", {})
        print(
            f"  Corpus: {meta.get('document_count')} files, "
            f"{meta.get('total_chars')} chars"
            + (" (truncated)" if meta.get("truncated") else "")
        )
        print("  Full-doc LLM metrics:")
        for k, v in full_payload["metrics"].items():
            print(f"    {k}: {v}")
        tu = eval_result.get("token_usage", {})
        print("\n  Token comparison (RAG vs Full-Chart):")
        print(f"    RAG total tokens: {tu.get('rag_tokens', '—')}")
        print(f"    Full-Chart total tokens: {tu.get('full_chart_tokens', '—')}")
        print(
            f"    Full-Chart additional vs RAG: "
            f"{tu.get('full_chart_additional_tokens', '—')} tokens "
            f"({tu.get('full_chart_additional_pct_vs_rag', '—')}% vs RAG)"
        )
        print(
            f"    Full-Chart share of question tokens: "
            f"{tu.get('full_chart_pct_of_question_tokens', '—')}% "
            f"(RAG {tu.get('rag_pct_of_question_tokens', '—')}%, "
            f"question total {tu.get('question_total_tokens', '—')})"
        )
        print("\n  --- FULL-CHART PHYSICIAN ANSWER ---")
        print(full_payload["answer"])

        # Step 3c: Judge
        print("\n  >>> STEP C — JUDGE ASSESSMENT (RAG vs FULL-CHART)")
        scores = judge_payload.get("scores", {})
        print(f"  Judge model: {judge_payload.get('judge_model')}")
        print(
            f"  Scores — Accuracy: {scores.get('accuracy_score')}  "
            f"Quality: {scores.get('quality_score')}  "
            f"Thoroughness: {scores.get('thoroughness_score')}  "
            f"Global accuracy: {scores.get('global_accuracy_score')}"
        )
        for field in (
            "accuracy_assessment",
            "quality_assessment",
            "thoroughness_assessment",
            "overall_summary",
        ):
            if scores.get(field):
                print(f"\n  {field}:")
                print(f"    {scores[field]}")
        print()

    results["steps"].append({"qa": qa_results})
    print_score_matrix(qa_results)
    results["total_elapsed_sec"] = round(time.perf_counter() - pipeline_start, 2)

    separator("PIPELINE COMPLETE")
    results["score_matrix"] = [
        {
            "id": item["id"],
            **item.get("judge", {}).get("scores", {}),
            **item.get("token_usage", {}),
        }
        for item in qa_results
    ]

    # Save machine-readable report
    report_path = config.DMS_ROOT / "last_run_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull report written to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""CLI: ingest documents or ask questions with Ollama RAG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from ingest import run_ingest
from ollama_client import check_ollama
from query import ask


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RAG over docs/ using Ollama (Docker) + Chroma"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Load docs/, chunk, embed, store vectors")

    ask_p = sub.add_parser("ask", help="Ask one question against the index")
    ask_p.add_argument("question", nargs="+", help="Question text")

    chat_p = sub.add_parser("chat", help="Interactive Q&A session")
    chat_p.add_argument(
        "--reingest",
        action="store_true",
        help="Re-run ingest before starting chat",
    )

    eval_p = sub.add_parser("eval", help="Run benchmark experiments (see experiments/)")
    eval_p.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/benchmark.yaml"),
        help="Benchmark YAML config",
    )
    eval_p.add_argument("--only", dest="only_experiment", default=None)
    eval_p.add_argument("--dry-run", action="store_true", help="One question only")
    eval_p.add_argument("--all-questions", action="store_true")
    eval_p.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Reuse existing Chroma collections",
    )

    args = parser.parse_args()

    if args.command == "eval":
        from eval.benchmark import run_benchmark

        if not config.DOCS_DIR.is_dir():
            print(f"Docs folder not found: {config.DOCS_DIR}", file=sys.stderr)
            return 1
        try:
            check_ollama()
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        run_benchmark(
            args.config,
            only_experiment=args.only_experiment,
            dry_run=args.dry_run,
            all_questions=args.all_questions,
            skip_ingest=args.skip_ingest,
        )
        return 0

    if not config.DOCS_DIR.is_dir():
        print(f"Docs folder not found: {config.DOCS_DIR}", file=sys.stderr)
        return 1

    try:
        check_ollama()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.command == "ingest":
        run_ingest(clear=True)
        return 0

    if args.command == "ask":
        question = " ".join(args.question)
        print(ask(question))
        return 0

    if args.command == "chat":
        if args.reingest:
            run_ingest(clear=True)
        else:
            coll = config.collection_name_for(
                "fixed_chars", config.OLLAMA_EMBED_MODEL
            )
            from store import get_collection

            if get_collection(collection_name=coll).count() == 0:
                print("No index found - running ingest first...")
                run_ingest(clear=True)

        print("RAG chat (empty line or 'quit' to exit)\n")
        while True:
            try:
                line = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line.lower() in ("quit", "exit", "q"):
                break
            print(f"\nAssistant: {ask(line)}\n")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

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

    args = parser.parse_args()

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
        elif not (config.CHROMA_DIR / "chroma.sqlite3").exists():
            print("No index found running ingest first...")
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

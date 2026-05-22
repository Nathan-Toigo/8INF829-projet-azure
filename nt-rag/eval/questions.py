"""Load benchmark questions from JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_questions(path: Path, *, all_questions: bool = False) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions", data)
    if not isinstance(questions, list):
        raise ValueError(f"Invalid questions file: {path}")

    if all_questions:
        return questions

    subset = data.get("default_subset")
    if subset:
        ids = set(subset)
        filtered = [q for q in questions if q.get("id") in ids]
        if filtered:
            return filtered
    return questions

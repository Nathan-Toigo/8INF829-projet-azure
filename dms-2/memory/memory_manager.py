"""Memory layer for the clinical agent (distinct from the agent graph).

- Short-term: a rolling window of the most recent conversation turns / step
  notes, passed each turn.
- Long-term: durable, summarized patient facts persisted to disk per patient and
  retrieved by relevance. Summarization is delegated to a caller-provided
  callable (wired to the ``summarize_history`` MCP tool in the graph), so this
  module stays decoupled from the tool transport.

``evaluate_and_store`` is the routing entry point: it decides whether new
information belongs in short-term or long-term memory.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

# Steps whose outputs represent durable patient knowledge worth long-term storage.
_DURABLE_STEPS = {"context", "risks", "synthesis"}
_LONG_TERM_MIN_CHARS = 200


@dataclass
class MemoryRecord:
    kind: str  # "short" | "long"
    step: str
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryManager:
    def __init__(self, patient_id: str, summarizer=None):
        self.patient_id = patient_id
        self.summarizer = summarizer  # Callable[[str, str], str] | None
        self.short_term: list[MemoryRecord] = []
        self.long_term: list[MemoryRecord] = []
        self._store_path = config.MEMORY_STORE_DIR / f"{self._safe_id()}.json"
        self._load()

    def _safe_id(self) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in self.patient_id)

    def _load(self) -> None:
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                self.long_term = [MemoryRecord(**r) for r in data.get("long_term", [])]
            except Exception:
                self.long_term = []

    def _persist(self) -> None:
        config.MEMORY_STORE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "patient_id": self.patient_id,
            "long_term": [r.to_dict() for r in self.long_term],
        }
        self._store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- short-term ---
    def add_turn(self, role: str, content: str) -> None:
        self.short_term.append(MemoryRecord(kind="short", step=role, content=content))
        max_n = config.SHORT_TERM_MAX_TURNS
        if len(self.short_term) > max_n:
            self.short_term = self.short_term[-max_n:]

    # --- routing ---
    def should_store_long_term(self, step: str, text: str) -> bool:
        """Heuristic deciding whether a step output is durable patient knowledge."""
        text = (text or "").strip()
        if not text:
            return False
        return step in _DURABLE_STEPS or len(text) >= _LONG_TERM_MIN_CHARS

    def evaluate_and_store(
        self, step: str, text: str, precomputed_summary: str | None = None
    ) -> dict:
        """Route a step output to short- and/or long-term memory.

        ``precomputed_summary`` lets the async graph node supply a summary built
        via the ``summarize_history`` MCP tool. Returns a decision record for the
        audit panel.
        """
        text = (text or "").strip()
        decision = {"step": step, "to_short_term": False, "to_long_term": False}
        if not text:
            return decision

        # Working memory always gets the latest step note.
        self.add_turn(role=f"step:{step}", content=text)
        decision["to_short_term"] = True

        if self.should_store_long_term(step, text):
            summary = (precomputed_summary or "").strip()
            if not summary and self.summarizer is not None:
                try:
                    summary = (self.summarizer(text, step) or "").strip()
                except Exception:
                    summary = ""
            if not summary:
                summary = text
            self.long_term.append(
                MemoryRecord(kind="long", step=step, content=summary)
            )
            self._persist()
            decision["to_long_term"] = True
            decision["summary_preview"] = summary[:200]
        return decision

    # --- retrieval ---
    def recent_short_term(self, limit: int | None = None) -> list[MemoryRecord]:
        if limit is None:
            return list(self.short_term)
        return self.short_term[-limit:]

    def relevant_long_term(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        """Lightweight keyword-overlap ranking over stored long-term facts."""
        if not self.long_term:
            return []
        terms = {t for t in query.lower().split() if len(t) > 3}
        if not terms:
            return self.long_term[-limit:]
        scored = []
        for rec in self.long_term:
            words = set(rec.content.lower().split())
            scored.append((len(terms & words), rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for score, rec in scored[:limit]]

    def snapshot(self) -> dict:
        """Serializable view of memory state for the UI audit panel."""
        return {
            "patient_id": self.patient_id,
            "short_term": [r.to_dict() for r in self.short_term],
            "long_term": [r.to_dict() for r in self.long_term],
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
        }

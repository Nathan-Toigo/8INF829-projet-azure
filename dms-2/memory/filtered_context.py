"""Token-budgeted context assembly from short-term, long-term, and RAG sources.

``FilteredContext`` ranks and trims the three context sources to fit a token
budget before prompt construction, implementing the
MemoryManager -> FilteredContext -> LLM step of the required data flow.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from memory.memory_manager import MemoryManager


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) to keep the layer dependency-free."""
    return max(1, len(text) // 4)


@dataclass
class AssembledContext:
    text: str
    token_estimate: int
    sources_used: dict

    def to_dict(self) -> dict:
        return {
            "token_estimate": self.token_estimate,
            "sources_used": self.sources_used,
            "preview": self.text[:600],
        }


class FilteredContext:
    def __init__(self, memory: MemoryManager, budget: int | None = None):
        self.memory = memory
        self.budget = budget or config.CONTEXT_TOKEN_BUDGET

    def assemble(self, query: str, rag_hits: list[dict] | None = None) -> AssembledContext:
        rag_hits = rag_hits or []
        remaining = self.budget
        blocks: list[str] = []
        used = {"long_term": 0, "short_term": 0, "rag": 0}

        # 1) Long-term durable facts (highest priority, most relevant first).
        lt = self.memory.relevant_long_term(query, limit=5)
        if lt:
            section = ["## Long-term patient memory"]
            for rec in lt:
                cost = estimate_tokens(rec.content)
                if cost > remaining:
                    continue
                section.append(f"- ({rec.step}) {rec.content}")
                remaining -= cost
                used["long_term"] += 1
            if len(section) > 1:
                blocks.append("\n".join(section))

        # 2) Retrieved chart excerpts (RAG) with provenance.
        if rag_hits:
            section = ["## Retrieved chart excerpts"]
            for hit in rag_hits:
                snippet = f"[{hit.get('source')} | sim {hit.get('similarity')}] {hit.get('content', '')}"
                cost = estimate_tokens(snippet)
                if cost > remaining:
                    snippet = snippet[: remaining * 4]
                    cost = estimate_tokens(snippet)
                if cost <= 0:
                    break
                section.append(f"- {snippet}")
                remaining -= cost
                used["rag"] += 1
                if remaining <= 0:
                    break
            if len(section) > 1:
                blocks.append("\n".join(section))

        # 3) Recent working memory (short-term), most recent first within budget.
        st = self.memory.recent_short_term()
        if st:
            section = ["## Recent working memory"]
            for rec in reversed(st):
                line = f"- ({rec.step}) {rec.content[:400]}"
                cost = estimate_tokens(line)
                if cost > remaining:
                    continue
                section.append(line)
                remaining -= cost
                used["short_term"] += 1
            if len(section) > 1:
                blocks.append("\n".join(section))

        text = "\n\n".join(blocks)
        return AssembledContext(
            text=text, token_estimate=estimate_tokens(text), sources_used=used
        )

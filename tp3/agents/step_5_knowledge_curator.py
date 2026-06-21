"""5.2 Knowledge Curator Agent - decide what is worth remembering.

Compares the (de-identified) case against the existing knowledge base and
decides whether it is novel and generalizable enough to be useful for future,
unrelated cases. It only commits to the official record once the Step-5
consensus is reached - Compliance approved the anonymization AND Reflection
judged the run trustworthy AND the Curator itself approves (flag-based gate).

On commit it calls ``tools.knowledge_store.store_knowledge`` which writes the
record to the MongoDB ``curated_knowledge`` collection (system of record) and
indexes it into the ``agent_learnings`` ChromaDB collection for semantic recall.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents.base_agent import AgentResponse, BaseAgent
from config import settings
from core import llm
from tools import knowledge_store

AGENT_ID = "5.2 Knowledge Curator Agent"
LIKELY_NEXT = ["1.1 Clinical Agent Orchestrator"]


class KnowledgeCuratorOutput(BaseModel):
    is_useful: bool = Field(
        default=False,
        description="True if this case yields generalizable knowledge useful "
                    "for future, different patients.",
    )
    knowledge_title: str = Field(default="", description="Short title for the learning.")
    knowledge_text: str = Field(
        default="",
        description="De-identified, generalizable lesson/pattern - NO patient "
                    "identifiers, written to help future unrelated cases.",
    )
    tags: list[str] = Field(default_factory=list)
    supporting_agents: list[str] = Field(
        default_factory=list,
        description="Which agents/evidence support this learning.",
    )
    decision_reason: str = Field(default="", description="Why store or skip this knowledge.")
    next_agent: str = Field(default="1.1 Clinical Agent Orchestrator")
    handoff_reason: str = ""
    needs_orchestrator: bool = True


_SYSTEM = (
    "You are the 5.2 Knowledge Curator Agent in an autonomous clinical "
    "multi-agent system. Using ONLY the de-identified case summary provided, "
    "decide whether this case contains a novel, generalizable lesson that would "
    "help future, unrelated patients - not a patient-specific fact. You are "
    "shown the most similar existing knowledge so you can judge novelty. "
    "Guardrails: store ONLY de-identified, generalizable knowledge; NEVER "
    "include patient identifiers or copy the raw chart; cite which agents/evidence "
    "support the learning; if the case merely repeats existing knowledge or is "
    "too patient-specific, do not store it and explain why. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}; after curating, "
    "return to the orchestrator."
)


def _candidate_text(state: dict) -> str:
    sanitized = state.get("sanitized_case") or {}
    summary = (sanitized.get("summary") if isinstance(sanitized, dict) else "") or ""
    return summary.strip()


class KnowledgeCuratorAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "strong"

    def _defer(self, reason: str) -> tuple[AgentResponse, list, list]:
        return (
            AgentResponse(
                agent_id=AGENT_ID,
                status="blocked",
                memory_updates={},
                next_agent=None,
                handoff_reason=reason,
                needs_orchestrator=True,
            ),
            [],
            [],
        )

    def execute(self, state):
        # Consensus prerequisites: Compliance and Reflection must have run.
        if not state.get("compliance_reviewed"):
            return self._defer(
                "Compliance review has not run yet; deferring so the case is "
                "de-identified before curation."
            )
        if not state.get("reflection_done"):
            return self._defer(
                "Reflection has not run yet; deferring until the run is judged."
            )

        candidate = _candidate_text(state)
        compliance_ok = bool(state.get("compliance_approved"))
        reflection_ok = bool(state.get("reflection_approved"))

        # Novelty from the existing curated knowledge (1 - max similarity).
        similar = knowledge_store.find_similar_knowledge(candidate, k=5) if candidate else []
        max_sim = max((h.get("similarity", 0.0) for h in similar), default=0.0)
        novelty = round(1.0 - max_sim, 4)

        similar_block = "\n".join(
            f"- (sim {h.get('similarity', 0):.2f}) {str(h.get('content', ''))[:200]}"
            for h in similar
        ) or "(no similar existing knowledge found)"

        user = (
            f"{self.context_block(state)}\n\n"
            f"DE-IDENTIFIED CASE SUMMARY:\n{candidate or '(none provided)'}\n\n"
            f"MOST SIMILAR EXISTING KNOWLEDGE:\n{similar_block}\n\n"
            f"Computed novelty score (1 - max similarity): {novelty}\n"
            f"Compliance approved: {compliance_ok}; Reflection approved: {reflection_ok}.\n\n"
            "Decide whether this is novel, generalizable knowledge worth storing."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=[],
            schema=KnowledgeCuratorOutput,
            tier=self.TIER,
            temperature=0.1,
        )

        curator_approved = bool(
            parsed.is_useful
            and parsed.knowledge_text.strip()
            and novelty >= settings.KNOWLEDGE_NOVELTY_THRESHOLD
        )

        # Flag-based consensus gate.
        consensus = compliance_ok and reflection_ok and curator_approved
        committed = False
        curated_id = ""
        decision_reason = parsed.decision_reason

        if consensus and candidate:
            try:
                result = knowledge_store.store_knowledge(
                    {
                        "text": parsed.knowledge_text,
                        "title": parsed.knowledge_title,
                        "tags": parsed.tags,
                        "patientId": state.get("patient_id"),
                        "runId": state.get("run_id"),
                        "supportingAgents": parsed.supporting_agents,
                        "noveltyScore": novelty,
                    }
                )
                curated_id = result.get("mongo_id") or ""
                committed = bool(curated_id)
                decision_reason = (
                    parsed.decision_reason
                    or "Novel, de-identified, consensus-approved learning committed."
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                decision_reason = f"Commit failed: {exc}"
        elif not decision_reason:
            if not consensus:
                blockers = []
                if not compliance_ok:
                    blockers.append("compliance not approved")
                if not reflection_ok:
                    blockers.append("reflection not approved")
                if not curator_approved:
                    blockers.append(
                        f"novelty {novelty} < {settings.KNOWLEDGE_NOVELTY_THRESHOLD} "
                        "or not generalizable"
                    )
                decision_reason = "Not committed: " + "; ".join(blockers) + "."
            else:
                decision_reason = "Not committed: no de-identified case available."

        memory_updates = {
            "knowledge_candidate": {
                "title": parsed.knowledge_title,
                "text": parsed.knowledge_text,
                "tags": parsed.tags,
                "supportingAgents": parsed.supporting_agents,
            },
            "knowledge_novelty_score": novelty,
            "curator_approved": curator_approved,
            "knowledge_committed": committed,
            "knowledge_decision_reason": decision_reason,
            "knowledge_curation_done": True,
        }
        if curated_id:
            memory_updates["curated_knowledge_id"] = curated_id

        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed",
            memory_updates=memory_updates,
            next_agent=None,
            handoff_reason=parsed.handoff_reason
            or (
                "Knowledge committed to the official record; returning to orchestrator."
                if committed
                else "No new knowledge committed; returning to orchestrator."
            ),
            needs_orchestrator=True,
        )
        return response, tool_records, token_records


run = KnowledgeCuratorAgent()

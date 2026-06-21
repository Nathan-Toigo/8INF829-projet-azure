"""5.3 Reflection Agent - LLM-as-a-Judge over the whole run.

Reviews every agent that contributed to the run and assesses whether each one
actually helped solve the case. The final client-facing output (the patient
explanation + key points + recommended actions) is the baseline: any agent whose
contribution is not aligned with / useful to that final answer is deemed unfit
and is flagged for prompt improvement.

It logs ONE MongoDB ``agent_evaluations`` entry per run containing every agent's
score, the below-threshold (flagged) agents, the root cause of each
misalignment, and concrete improvement ideas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents.base_agent import AgentResponse, BaseAgent
from config import settings
from core import llm
from tools import mongodb_tools

AGENT_ID = "5.3 Reflection Agent"
LIKELY_NEXT = [
    "5.2 Knowledge Curator Agent",
    "1.1 Clinical Agent Orchestrator",
]

# Agents excluded from judging (the supervisor and the learning-layer agents
# that do not contribute clinical content to the final answer).
_NON_SCORED = {
    "1.1 Clinical Agent Orchestrator",
    "5.1 Compliance PII Agent",
    "5.3 Reflection Agent",
    "5.2 Knowledge Curator Agent",
}

# Overall-alignment bar (independent of the per-agent flag threshold) used to
# decide whether the run's reasoning is trustworthy enough to curate.
_APPROVAL_MIN_AVERAGE = 0.6


class AgentScoreItem(BaseModel):
    agent: str = ""
    score: float = Field(default=0.0, description="0-1 alignment/usefulness vs the final answer.")
    aligned: bool = Field(default=False, description="Did this agent meaningfully help the final answer?")
    rationale: str = ""


class AgentFinding(BaseModel):
    agent: str = ""
    root_cause: str = Field(default="", description="Why this agent was misaligned/unhelpful.")
    improvements: list[str] = Field(
        default_factory=list,
        description="Concrete prompt/behaviour improvement ideas for this agent.",
    )


class ReflectionOutput(BaseModel):
    scores: list[AgentScoreItem] = Field(default_factory=list)
    findings: list[AgentFinding] = Field(
        default_factory=list,
        description="Root cause + improvements for any agent judged misaligned "
                    "or weak (score below ~0.7 or not aligned).",
    )
    summary: str = Field(default="", description="Short overall reflection on the run.")
    next_agent: str = Field(default="5.2 Knowledge Curator Agent")
    handoff_reason: str = ""
    needs_orchestrator: bool = False


_SYSTEM = (
    "You are the 5.3 Reflection Agent, acting as an impartial LLM-as-a-Judge over "
    "an autonomous clinical multi-agent run. The BASELINE is the final "
    "client-facing answer (patient explanation + key points + recommended "
    "actions). For EACH agent that ran, score 0-1 how well its contribution is "
    "aligned with and useful to that final answer, mark whether it meaningfully "
    "helped, and give a brief rationale. Any agent not aligned with the final "
    "answer is deemed unfit: provide a finding with the root cause of the "
    "misalignment and concrete ideas to improve that agent's prompt. "
    "Guardrails: judge ONLY on the evidence provided in shared memory; do not "
    "fabricate scores or contributions; be consistent and calibrated. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}."
)


def _baseline(state: dict) -> str:
    parts = [f"FINAL ANSWER:\n{state.get('patient_explanation', '')}"]
    kp = state.get("patient_key_points") or []
    if kp:
        parts.append("KEY POINTS:\n" + "\n".join(f"- {p}" for p in kp))
    ra = state.get("patient_recommended_actions") or []
    if ra:
        parts.append("RECOMMENDED ACTIONS:\n" + "\n".join(f"- {a}" for a in ra))
    return "\n\n".join(parts)


def _scored_agents(state: dict) -> list[str]:
    seen: list[str] = []
    for a in state.get("agents_run", []):
        if a not in _NON_SCORED and a not in seen:
            seen.append(a)
    return seen


class ReflectionAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "strong"

    def execute(self, state):
        baseline = state.get("patient_explanation", "").strip()
        if not baseline:
            return (
                AgentResponse(
                    agent_id=AGENT_ID,
                    status="blocked",
                    memory_updates={},
                    next_agent=None,
                    handoff_reason=(
                        "No final patient answer to use as a baseline yet; "
                        "deferring to orchestrator to produce it first."
                    ),
                    needs_orchestrator=True,
                ),
                [],
                [],
            )

        agents = _scored_agents(state)
        user = (
            f"{self.context_block(state)}\n\n"
            f"{_baseline(state)}\n\n"
            f"AGENTS THAT RAN (score each of these): {agents}\n\n"
            "Score every listed agent against the final answer and provide "
            "findings for any misaligned/weak agent now."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=[],
            schema=ReflectionOutput,
            tier=self.TIER,
            temperature=0.0,
        )

        scores = [s.model_dump() for s in parsed.scores]
        valid = [max(0.0, min(1.0, s["score"])) for s in scores]
        average = round(sum(valid) / len(valid), 4) if valid else 0.0

        configured = settings.REFLECTION_SCORE_THRESHOLD
        threshold = round(configured if configured > 0 else average, 4)

        findings_by_agent = {f.agent: f for f in parsed.findings}
        flagged = []
        for s in scores:
            if s["score"] < threshold:
                f = findings_by_agent.get(s["agent"])
                flagged.append(
                    {
                        "agent": s["agent"],
                        "score": s["score"],
                        "rootCause": f.root_cause if f else "",
                        "improvements": f.improvements if f else [],
                    }
                )

        reflection_approved = bool(valid) and average >= _APPROVAL_MIN_AVERAGE

        evaluation = {
            "runId": state.get("run_id"),
            "patientId": state.get("patient_id"),
            "question": state.get("patient_question", ""),
            "baseline": _baseline(state),
            "threshold": threshold,
            "average": average,
            "scores": scores,
            "flagged": flagged,
            "summary": parsed.summary,
            "approved": reflection_approved,
        }
        eval_id = None
        try:
            eval_id = mongodb_tools.insert_agent_evaluation(evaluation)
        except Exception:
            eval_id = None

        memory_updates = {
            "agent_scores": scores,
            "reflection_average": average,
            "reflection_threshold": threshold,
            "reflection_flagged": flagged,
            "reflection_findings": [f.model_dump() for f in parsed.findings],
            "reflection_approved": reflection_approved,
            "reflection_done": True,
            "reflection_evaluation_id": eval_id or "",
        }

        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed",
            memory_updates=memory_updates,
            next_agent=None if parsed.needs_orchestrator else parsed.next_agent,
            handoff_reason=parsed.handoff_reason
            or (
                f"Scored {len(scores)} agent(s); avg {average}, "
                f"{len(flagged)} flagged. Knowledge curation next."
            ),
            needs_orchestrator=parsed.needs_orchestrator,
        )
        return response, tool_records, token_records


run = ReflectionAgent()

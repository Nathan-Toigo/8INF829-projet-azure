"""4.3 Clinical Review Agent - global reviewer of the produced output.

Acts as a senior clinician auditing the whole multi-agent run before returning
the answer. Cross-checks the patient-facing explanation against the timeline,
guidelines, risks, similar cases, and any hypotheses/care plan to flag clinical
inconsistencies, unsupported claims, or missing safety points (e.g. red flags
that did not surface in the patient explanation).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents.base_agent import AgentResponse, BaseAgent
from core import llm

AGENT_ID = "4.3 Clinical Review Agent"
LIKELY_NEXT = [
    "4.1 Patient Explanation Agent",
    "1.1 Clinical Agent Orchestrator",
]


class ClinicalReviewOutput(BaseModel):
    review_passed: bool = Field(
        default=False,
        description="True if the explanation is clinically sound and complete.",
    )
    clinical_score: float = Field(
        default=0.5,
        description="0-1 overall clinical quality score.",
    )
    inconsistencies: list[str] = Field(
        default_factory=list,
        description="Claims that contradict timeline/guidelines/risks/similar cases.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims not backed by any upstream agent output.",
    )
    missing_safety_points: list[str] = Field(
        default_factory=list,
        description="Red flags or critical follow-ups that should be surfaced to "
                    "the patient but are missing.",
    )
    overall_assessment: str = Field(
        default="",
        description="2-4 sentence clinical assessment of the produced output.",
    )
    next_agent: str = Field(default="1.1 Clinical Agent Orchestrator")
    handoff_reason: str = ""
    needs_orchestrator: bool = True


_SYSTEM = (
    "You are the 4.3 Clinical Review Agent, the senior clinician reviewer of "
    "the autonomous clinical multi-agent system. Audit the patient explanation "
    "currently in shared memory against ALL upstream evidence (timeline, "
    "guidelines, risks, red flags, similar cases, and any hypotheses or care "
    "plan). Identify clinical inconsistencies, unsupported claims, and missing "
    "safety points. Do not rewrite the explanation yourself; instead, decide "
    "whether it passes review and provide specific, actionable findings. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}. If the review fails "
    "and a fix is feasible, route back to 4.1; otherwise return to the "
    "orchestrator."
)


class ClinicalReviewAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "strong"

    def execute(self, state):
        explanation = state.get("patient_explanation", "")
        if not explanation:
            return (
                AgentResponse(
                    agent_id=AGENT_ID,
                    status="blocked",
                    memory_updates={},
                    next_agent=None,
                    handoff_reason=(
                        "No patient explanation available to review; deferring "
                        "to orchestrator."
                    ),
                    needs_orchestrator=True,
                ),
                [],
                [],
            )

        user = (
            f"{self.context_block(state)}\n\n"
            "Perform a senior clinician's review of the current patient_explanation "
            "against all upstream evidence in memory now."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=[],
            schema=ClinicalReviewOutput,
            tier=self.TIER,
        )

        # Compteur de re-essais pour eviter une boucle infinie de revision
        review_attempts = state.get("clinical_review_attempts", 0) + 1
        MAX_REVIEW_ATTEMPTS = 2

        needs_fix = (
            not parsed.review_passed
            and review_attempts < MAX_REVIEW_ATTEMPTS
            and (
                parsed.inconsistencies
                or parsed.unsupported_claims
                or parsed.missing_safety_points
            )
        )

        memory_updates = {
            "clinical_review_passed": parsed.review_passed,
            "clinical_score": parsed.clinical_score,
            "clinical_review_inconsistencies": parsed.inconsistencies,
            "clinical_review_unsupported_claims": parsed.unsupported_claims,
            "clinical_review_missing_safety_points": parsed.missing_safety_points,
            "clinical_review_assessment": parsed.overall_assessment,
            "clinical_review_attempts": review_attempts,
        }

        next_agent = (
            "4.1 Patient Explanation Agent"
            if needs_fix and not parsed.needs_orchestrator
            else (None if parsed.needs_orchestrator else parsed.next_agent)
        )

        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed",
            memory_updates=memory_updates,
            next_agent=next_agent,
            handoff_reason=parsed.handoff_reason
            or (
                "Clinical review passed; returning to orchestrator."
                if parsed.review_passed
                else (
                    "Clinical review found issues; rerouting to Patient Explanation."
                    if needs_fix
                    else f"Max review attempts ({MAX_REVIEW_ATTEMPTS}) reached; returning to orchestrator with current best version."
                )
            ),
            needs_orchestrator=(not needs_fix),
        )
        return response, tool_records, token_records


run = ClinicalReviewAgent()
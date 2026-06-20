"""4.2 Patient Representative Agent - is the information appropriate for the patient?

Acts as the patient's advocate: reviews the draft explanation produced by the
Patient Explanation Agent and verifies that it is understandable, non-alarming,
respectful, and actionable. Flags jargon, missing context, or anxiety-inducing
wording, and proposes a revised explanation when needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents.base_agent import AgentResponse, BaseAgent
from core import llm

AGENT_ID = "4.2 Patient Representative Agent"
LIKELY_NEXT = [
    "4.3 Clinical Review Agent",
    "4.1 Patient Explanation Agent",
    "1.1 Clinical Agent Orchestrator",
]


class PatientRepresentativeOutput(BaseModel):
    is_appropriate: bool = Field(
        default=False,
        description="True if the explanation is patient-appropriate as-is.",
    )
    appropriateness_score: float = Field(
        default=0.5,
        description="0-1 score of patient-appropriateness.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific issues found (jargon, anxiety, missing actions, etc.).",
    )
    suggested_improvements: list[str] = Field(default_factory=list)
    revised_explanation: str = Field(
        default="",
        description="Revised plain-language explanation if a rewrite is needed; "
                    "empty if the original is already appropriate.",
    )
    next_agent: str = Field(default="4.3 Clinical Review Agent")
    handoff_reason: str = ""
    needs_orchestrator: bool = False


_SYSTEM = (
    "You are the 4.2 Patient Representative Agent in an autonomous clinical "
    "multi-agent system. You act as an advocate for the patient. Review the "
    "draft patient explanation in shared memory and decide whether it is truly "
    "appropriate for a non-clinician: understandable wording (no unexplained "
    "jargon), non-alarming tone, culturally respectful, actionable, and faithful "
    "to the underlying clinical reasoning. If issues are present, list them "
    "concretely and propose a revised explanation. Never introduce new clinical "
    "facts that are not already supported by upstream memory. "
    f"Decide the next agent from: {', '.join(LIKELY_NEXT)}. If blocked, set "
    "needs_orchestrator=true and next_agent=null."
)


class PatientRepresentativeAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "small"

    def execute(self, state):
        draft = state.get("patient_explanation", "")
        if not draft:
            return (
                AgentResponse(
                    agent_id=AGENT_ID,
                    status="blocked",
                    memory_updates={},
                    next_agent=None,
                    handoff_reason=(
                        "No patient explanation drafted yet; deferring to "
                        "orchestrator to route to 4.1 first."
                    ),
                    needs_orchestrator=True,
                ),
                [],
                [],
            )

        user = (
            f"{self.context_block(state)}\n\n"
            "Review the current patient_explanation in memory and judge whether "
            "it is appropriate for the patient. Propose a revised version only "
            "if material improvements are needed."
        )
        parsed, tool_records, token_records = llm.run_agentic_step(
            step=AGENT_ID,
            system=_SYSTEM,
            user=user,
            tools=[],
            schema=PatientRepresentativeOutput,
            tier=self.TIER,
        )

        # L'explanation finale a afficher: la revision si elle existe et a remplace
        # le draft, sinon le draft original.
        final_explanation = (
            parsed.revised_explanation
            if (parsed.revised_explanation and not parsed.is_appropriate)
            else draft
        )

        memory_updates = {
            "patient_appropriateness_score": parsed.appropriateness_score,
            "patient_appropriateness_issues": parsed.issues,
            "patient_appropriateness_suggestions": parsed.suggested_improvements,
            "patient_appropriateness_passed": parsed.is_appropriate,
            "patient_friendly_explanation": final_explanation,  # alias pour Care Plan page
        }

        # Si l'agent fournit une revision, on la promeut comme nouvelle explanation
        # (et on garde la precedente pour tracabilite).
        if parsed.revised_explanation and not parsed.is_appropriate:
            memory_updates["patient_explanation_previous"] = state.get(
                "patient_explanation", ""
            )
            memory_updates["patient_explanation"] = parsed.revised_explanation

        response = AgentResponse(
            agent_id=AGENT_ID,
            status="completed",
            memory_updates=memory_updates,
            next_agent=None if parsed.needs_orchestrator else parsed.next_agent,
            handoff_reason=parsed.handoff_reason
            or (
                "Explanation is patient-appropriate; clinical review next."
                if parsed.is_appropriate
                else "Explanation revised for patient-appropriateness; clinical review next."
            ),
            needs_orchestrator=parsed.needs_orchestrator,
        )
        return response, tool_records, token_records


run = PatientRepresentativeAgent()
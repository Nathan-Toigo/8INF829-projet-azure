"""1.1 Clinical Agent Orchestrator - LLM-driven supervisor of the agent team.

Routing is primarily decided by an LLM that is given a definition of every
available agent, ground rules for navigating the four steps (gather -> reason ->
simplify -> improve), the shared working-memory summary, and a per-step phase
status. The LLM picks the next agent so findings keep accumulating in working
memory and the team iterates across all steps toward the best answer.

Thin deterministic guardrails remain only to keep the workflow safe and
terminating: a max-step cap, a heuristic fallback when the LLM is unavailable or
returns an invalid choice, Step-3 internal-ordering correction, and a rule that
prevents ending before the full pipeline has run (spec section 13).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from agents.base_agent import AgentResponse, BaseAgent
from config import settings
from core import llm
from memory import short_term_memory

AGENT_ID = "1.1 Clinical Agent Orchestrator"
END = "END"

STEP_2_ORDER = [
    ("2.1 Timeline Agent", "timeline"),
    ("2.2 Guidelines Agent", "guidelines"),
    ("2.3 Risk Agent", "risks"),
    ("2.4 Case Investigator Agent", "similar_cases"),
]

STEP_3_AGENTS = {
    "3.5 Care Planning Agent",
    "3.1 Investigation Planning Agent",
    "3.2 Hypothesis Agent",
    "3.3 Evidence Validation Agent",
    "3.4 Gap Validation Agent",
    "3.6 Confidence Assessment Agent",
}

STEP_4_ORDER = [
    ("4.1 Patient Explanation Agent", "patient_explanation"),
    ("4.2 Patient Representative Agent", "patient_appropriateness_passed"),
    ("4.3 Clinical Review Agent", "clinical_review_assessment"),
]

# Step 5 (learning layer) runs last, in 5.1 -> 5.3 -> 5.2 order so that
# Compliance and Reflection set their approval flags before the Knowledge
# Curator's consensus-gated commit. Each key is an always-truthy "done" marker
# the agent sets on completion, which guarantees the heuristic terminates.
STEP_5_ORDER = [
    ("5.1 Compliance PII Agent", "compliance_reviewed"),
    ("5.3 Reflection Agent", "reflection_done"),
    ("5.2 Knowledge Curator Agent", "knowledge_curation_done"),
]

FOUNDATION_ORDER = STEP_2_ORDER + STEP_4_ORDER + STEP_5_ORDER

# Canonical forward progression used to FORCE the workflow ahead once it has
# iterated for too long: Step 2 -> Step 3 -> Step 4 -> Step 5, then END. Each
# agent is visited at most once in forced mode (already-run agents are skipped),
# which guarantees the run marches through and completes Step 5.
STEP_3_PROGRESSION = [
    "3.5 Care Planning Agent",
    "3.1 Investigation Planning Agent",
    "3.2 Hypothesis Agent",
    "3.3 Evidence Validation Agent",
    "3.4 Gap Validation Agent",
    "3.6 Confidence Assessment Agent",
]
PROGRESSION_ORDER = (
    [a for a, _ in STEP_2_ORDER]
    + STEP_3_PROGRESSION
    + [a for a, _ in STEP_4_ORDER]
    + [a for a, _ in STEP_5_ORDER]
)

# Human-readable definition of every agent the orchestrator can route to,
# grouped by step. Drives both the LLM routing prompt and the canonical set of
# routable agents.
AGENT_CATALOG: list[tuple[str, str]] = [
    # Step 2 - gather the maximum information about the patient.
    ("2.1 Timeline Agent",
     "Builds the patient's chronological medical history (events, dates, gaps) "
     "from the records."),
    ("2.2 Guidelines Agent",
     "Retrieves the clinical guidelines and best-practice recommendations "
     "relevant to the question."),
    ("2.3 Risk Agent",
     "Identifies high-risk issues, red flags, and safety concerns in the "
     "patient data."),
    ("2.4 Case Investigator Agent",
     "Finds similar historical/known cases and reusable patterns relevant to "
     "the question."),
    # Step 3 - reason: plan, hypothesize, investigate, validate, score.
    ("3.5 Care Planning Agent",
     "Builds the Step-3 investigation/reasoning action plan (goals, priorities, "
     "search queries). Usually the first Step-3 agent."),
    ("3.1 Investigation Planning Agent",
     "Runs web/literature searches to gather external clinical evidence for the "
     "plan."),
    ("3.2 Hypothesis Agent",
     "Generates and ranks all plausible diagnostic/explanatory hypotheses from "
     "the gathered context."),
    ("3.3 Evidence Validation Agent",
     "Challenges and validates each hypothesis against the evidence; flags "
     "contradictions and unsupported claims."),
    ("3.4 Gap Validation Agent",
     "Decides whether the hypotheses are sufficient or critical information is "
     "still missing."),
    ("3.6 Confidence Assessment Agent",
     "Scores overall confidence and produces the care-plan recommendations; may "
     "trigger another Step-3 attempt."),
    # Step 4 - rephrase and simplify for a non-medical client.
    ("4.1 Patient Explanation Agent",
     "Writes the final plain-language answer for the patient from the clinical "
     "reasoning."),
    ("4.2 Patient Representative Agent",
     "Acts as the patient's advocate; checks the explanation is clear, "
     "non-alarming, and actionable."),
    ("4.3 Clinical Review Agent",
     "Senior-clinician audit of the explanation against all evidence for safety "
     "and accuracy."),
    # Step 5 - improvement loop: document and find room for improvement.
    ("5.1 Compliance PII Agent",
     "De-identifies the case (HIPAA + Quebec Law 25) and scores reidentification "
     "risk before anything is learned."),
    ("5.3 Reflection Agent",
     "LLM-as-judge: scores how well every agent helped the final answer and logs "
     "flagged agents, root causes, and improvement ideas."),
    ("5.2 Knowledge Curator Agent",
     "Decides whether the case yields novel, generalizable knowledge and commits "
     "it to the official knowledge base."),
]

AVAILABLE_AGENTS = [a for a, _ in AGENT_CATALOG]


class OrchestratorDecision(BaseModel):
    intent: str = Field(default="", description="Classified intent of the question.")
    next_agent: str = Field(default="", description="Chosen next agent id, or 'END'.")
    reason: str = Field(default="", description="Why this agent is the best next step.")


_AGENT_DEFINITIONS = "\n".join(f"- {a}: {d}" for a, d in AGENT_CATALOG)

_GROUND_RULES = (
    "1. ALWAYS gather the maximum information before answering the patient: run "
    "the Step 2 agents (2.1-2.4) to build the timeline, pull guidelines, surface "
    "risks, and find similar cases.\n"
    "2. ALWAYS reason thoroughly with the Step 3 agents: plan the investigation "
    "(3.5), run investigations (3.1), generate ALL plausible hypotheses (3.2), "
    "gather and validate evidence (3.3), validate gaps (3.4), and assess "
    "confidence (3.6). Iterate the hypothesis/evidence loop as needed.\n"
    "3. ALWAYS rephrase and simplify the answer for a non-medical client with the "
    "Step 4 agents: draft the explanation (4.1), advocate for the patient (4.2), "
    "and clinically review it (4.3).\n"
    "4. ALWAYS go through the Step 5 improvement loop to document the run and find "
    "room for improvement: de-identify (5.1), reflect/score every agent (5.3), "
    "and curate reusable knowledge (5.2)."
)

_SYSTEM = (
    "You are the Clinical Agent Orchestrator, the supervisor of an autonomous "
    "clinical multi-agent system. You decide, step by step, which agent should "
    "run next so the team produces the best possible answer for the patient. "
    "Base every decision on the shared working memory (the state summary and "
    "phase status you are given), and route work so that findings keep "
    "accumulating in that memory.\n\n"
    "AVAILABLE AGENTS:\n" + _AGENT_DEFINITIONS + "\n\n"
    "GROUND RULES (follow in order, but iterate as much as needed):\n"
    + _GROUND_RULES + "\n\n"
    "ROUTING POLICY:\n"
    "- Navigate across ALL steps (2 -> 3 -> 4 -> 5) and iterate with as many "
    "agents as possible to converge on the best answer; you may revisit an agent "
    "when new information justifies it, but do not needlessly repeat an agent "
    "that already succeeded with no new input.\n"
    "- Move to the next step only once the current step's goal is met in memory.\n"
    "- For Step 3, prefer the provided heuristic suggestion for the internal "
    "ordering (its agents have strict prerequisites).\n"
    "- Only choose END after Step 4 has produced a clinically reviewed patient "
    "answer AND the Step 5 improvement loop has run.\n"
    "Choose next_agent strictly from the agent ids listed above, or 'END'."
)


def _step_2_complete(state: dict) -> bool:
    return all(short_term_memory.has_content(state, key) for _, key in STEP_2_ORDER)


def _step_3_next(state: dict) -> str | None:
    if state.get("step_3_complete"):
        return None

    if state.get("step_3_restart_requested"):
        return "3.5 Care Planning Agent"

    if not state.get("step_3_care_plan_done"):
        return "3.5 Care Planning Agent"
    if not state.get("step_3_investigation_done"):
        return "3.1 Investigation Planning Agent"
    if not short_term_memory.has_content(state, "hypotheses"):
        return "3.2 Hypothesis Agent"
    return None


def _heuristic_next(state: dict) -> str:
    for agent_id, key in STEP_2_ORDER:
        if not short_term_memory.has_content(state, key):
            return agent_id

    if _step_2_complete(state) and not state.get("step_3_complete"):
        step_3 = _step_3_next(state)
        if step_3:
            return step_3

    for agent_id, key in STEP_4_ORDER:
        if not short_term_memory.has_content(state, key):
            return agent_id

    for agent_id, key in STEP_5_ORDER:
        if not short_term_memory.has_content(state, key):
            return agent_id

    return END


def _forced_next(state: dict) -> str:
    """Next agent when forcing forward progression: the first agent in the
    canonical order that has not run yet, else END (Step 5 reached/complete)."""
    ran = set(state.get("agents_run", []))
    for agent_id in PROGRESSION_ORDER:
        if agent_id not in ran:
            return agent_id
    return END


def _phase_status(state: dict) -> str:
    """Compact per-step progress signal so the LLM can route across all steps."""

    def yn(flag) -> str:
        return "yes" if flag else "no"

    step2 = ", ".join(
        f"{key}={yn(short_term_memory.has_content(state, key))}"
        for _, key in STEP_2_ORDER
    )
    lines = [
        "PHASE STATUS:",
        f"- Step 2 (gather): {step2}; complete={yn(_step_2_complete(state))}",
        (
            "- Step 3 (reason): "
            f"care_plan_done={yn(state.get('step_3_care_plan_done'))}, "
            f"investigation_done={yn(state.get('step_3_investigation_done'))}, "
            f"hypotheses={len(state.get('hypotheses') or [])}, "
            f"hypotheses_sufficient={yn(state.get('hypotheses_sufficient'))}, "
            f"confidence={state.get('confidence_score', 0)}, "
            f"attempt={state.get('step_3_attempt', 1)}, "
            f"complete={yn(state.get('step_3_complete'))}"
        ),
        (
            "- Step 4 (simplify): "
            f"explanation={yn(state.get('patient_explanation'))}, "
            f"patient_appropriate={yn(state.get('patient_appropriateness_passed'))}, "
            f"clinically_reviewed={yn(state.get('clinical_review_assessment'))}"
        ),
        (
            "- Step 5 (improve): "
            f"compliance_reviewed={yn(state.get('compliance_reviewed'))}, "
            f"reflection_done={yn(state.get('reflection_done'))}, "
            f"knowledge_curation_done={yn(state.get('knowledge_curation_done'))}"
        ),
    ]
    return "\n".join(lines)


class OrchestratorAgent(BaseAgent):
    AGENT_ID = AGENT_ID
    TIER = "small"

    def execute(self, state):
        token_records: list = []

        if state.get("step_count", 0) >= settings.MAX_AGENT_STEPS:
            return (
                AgentResponse(
                    agent_id=AGENT_ID,
                    status="completed",
                    memory_updates={},
                    next_agent=END,
                    handoff_reason="Max agent steps reached; ending workflow.",
                    needs_orchestrator=False,
                ),
                [],
                [],
            )

        heuristic = _heuristic_next(state)
        intent = state.get("intent", "")
        step_count = state.get("step_count", 0)
        force_after = settings.FORCE_PROGRESSION_AFTER_STEPS
        forcing = force_after > 0 and step_count >= force_after

        # Once we are forcing progression, skip the LLM and march forward
        # deterministically through the steps until Step 5 is complete.
        if forcing:
            choice = _forced_next(state)
            reason = (
                f"Forced forward progression after {step_count} steps "
                f"(>= {force_after}): advancing to {choice} to complete the "
                "remaining steps through Step 5."
            )
            memory_updates: dict = {}
            if state.get("step_3_restart_requested") and choice == "3.5 Care Planning Agent":
                memory_updates["step_3_restart_requested"] = False
            next_agent = None if choice == END else choice
            return (
                AgentResponse(
                    agent_id=AGENT_ID,
                    status="completed",
                    memory_updates=memory_updates,
                    next_agent=next_agent,
                    handoff_reason=reason,
                    needs_orchestrator=False,
                ),
                [],
                token_records,
            )

        try:
            user = (
                f"{self.context_block(state)}\n\n"
                f"{_phase_status(state)}\n\n"
                f"Agents already run (in order): {state.get('agents_run', [])}\n"
                f"Heuristic suggestion (a safe default, especially for Step 3): {heuristic}\n\n"
                "Using the ground rules and routing policy, classify the intent "
                "and choose the single best next agent to run (or END)."
            )
            decision, usage = llm.invoke_structured(
                step=AGENT_ID,
                system=_SYSTEM,
                user=user,
                schema=OrchestratorDecision,
                tier=self.TIER,
            )
            token_records.append(usage)
            intent = decision.intent or intent
            choice = decision.next_agent.strip()
            reason = decision.reason
        except Exception:
            choice, reason = heuristic, "Heuristic routing (LLM unavailable)."

        # Guardrails around the LLM's decision (it remains the primary driver):
        # 1. Unknown/empty choice -> safe heuristic.
        if choice not in AVAILABLE_AGENTS and choice != END:
            choice = heuristic
            reason = reason or f"Invalid choice; falling back to {heuristic}."
        # 2. Step-3 agents have strict prerequisites; keep their internal order.
        elif choice in STEP_3_AGENTS:
            expected = _step_3_next(state)
            if expected and choice != expected:
                choice = expected
                reason = (
                    reason or ""
                ) + f" (corrected to {expected} to respect Step-3 ordering)"
        # 3. Do not END before the full pipeline has run (ground rule 4); if the
        #    heuristic still has required work queued, continue with it.
        if choice == END and heuristic != END:
            choice = heuristic
            reason = (
                f"Pipeline not complete yet; continuing with {heuristic} before END."
            )

        memory_updates: dict = {}
        if intent and intent != state.get("intent"):
            memory_updates["intent"] = intent
        if state.get("step_3_restart_requested") and choice == "3.5 Care Planning Agent":
            memory_updates["step_3_restart_requested"] = False

        next_agent = None if choice == END else choice
        return (
            AgentResponse(
                agent_id=AGENT_ID,
                status="completed",
                memory_updates=memory_updates,
                next_agent=next_agent,
                handoff_reason=reason or f"Routing to {choice}.",
                needs_orchestrator=False,
            ),
            [],
            token_records,
        )


run = OrchestratorAgent()

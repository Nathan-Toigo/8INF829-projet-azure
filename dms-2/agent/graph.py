"""LangGraph wiring of the 5-step clinical pipeline + memory-evaluation nodes.

Data flow per the plan:
Streamlit -> LangGraph -> MCP tools -> Agent -> MemoryManager -> FilteredContext
-> LLM thought iteration -> Update Memory.

The agent walks a fixed clinical pipeline (a deterministic ReAct variant): each
step proactively calls the MCP tools relevant to that step, assembles a
token-budgeted context, then produces a structured result. A memory-evaluation
node after each step routes findings to short- or long-term memory.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

import config
from agent import llm, prompts
from agent.mcp_client import ClinicalToolset, tool_result_to_json, tool_result_to_text
from agent.schemas import (
    ActionResult,
    ContextResult,
    PriorityResult,
    RiskResult,
    SynthesisResult,
)
from agent.state import AgentState
from memory.filtered_context import FilteredContext
from memory.memory_manager import MemoryManager


class ClinicalAgent:
    def __init__(
        self,
        patient_id: str,
        language: str,
        toolset: ClinicalToolset,
        memory: MemoryManager,
    ):
        self.patient_id = patient_id
        self.language = language
        self.toolset = toolset
        self.memory = memory
        self.filtered = FilteredContext(memory)
        self.graph = self._build()

    # ------------------------------------------------------------------ tools
    async def call_tool(self, step: str, tool_name: str, args: dict):
        tool = self.toolset.get(tool_name)
        t0 = time.perf_counter()
        if tool is None:
            result = f"[tool '{tool_name}' unavailable]"
        else:
            try:
                result = await tool.ainvoke(args)
            except Exception as exc:  # pragma: no cover - runtime guard
                result = f"[tool '{tool_name}' error: {exc}]"
        ms = (time.perf_counter() - t0) * 1000

        result_str = tool_result_to_text(result)
        parsed = tool_result_to_json(result)

        record = {
            "step": step,
            "tool": tool_name,
            "args": args,
            "latency_ms": round(ms, 2),
            "result_preview": result_str[:600],
        }
        return result_str, parsed, record

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _find_step(state: AgentState, step_id: str) -> dict:
        for r in reversed(state.get("step_results", [])):
            if r.get("step_id") == step_id:
                return r.get("structured", {})
        return {}

    # ------------------------------------------------------------------ nodes
    def _tools(self, names: list[str]) -> list:
        return [self.toolset.get(n) for n in names if self.toolset.get(n) is not None]

    def _memory_context(self, query: str):
        """Memory-only context (long + short term). The LLM digs further via tools."""
        return self.filtered.assemble(query, [])

    _TOOL_MENU = (
        "\n\nAvailable MCP tools you may call iteratively to dig deeper:\n"
        "- rag_search_documents(query, k): retrieve chart excerpts (with sources).\n"
        "- build_patient_timeline(query): chronological events.\n"
        "- analyze_lab_trends(query): longitudinal lab analysis.\n"
        "- check_drug_interactions(medications): interaction review.\n"
        "- web_search_tool(query): current guidelines / drug info.\n"
        "Call tools as many times as needed (different queries) to fill gaps in "
        "the memory context, then stop and produce the result."
    )

    async def node_context(self, state: AgentState) -> dict:
        step = "context"
        query = state.get("user_request") or (
            "overall clinical context, chronic conditions, current medications, history"
        )
        ctx = self._memory_context(query)
        user = (
            f"Patient request: {query}\n\n"
            f"## Existing memory context\n{ctx.text or '(empty - rely on tools)'}\n"
            "\nGather evidence with the tools (search the documents, build the "
            "timeline), then identify chronic conditions, current medications, and a "
            "longitudinal portrait. Cite source documents." + self._TOOL_MENU
        )
        parsed, tool_records, token_records = await llm.run_agentic_step(
            step,
            prompts.STEP_CONTEXT_SYSTEM,
            user,
            self._tools(["rag_search_documents", "build_patient_timeline"]),
            ContextResult,
        )
        output = self._format_context(parsed)
        return {
            "current_step": step,
            "step_results": [
                {
                    "step_id": step,
                    "title": "1. Comprendre le contexte",
                    "structured": parsed.model_dump(),
                    "output": output,
                    "context_used": ctx.to_dict(),
                }
            ],
            "tool_calls": tool_records,
            "token_ledger": token_records,
            "rag_available": any(
                tc["tool"] == "rag_search_documents" for tc in tool_records
            ),
            "pending_memory": {"step": step, "text": output},
            "messages": [AIMessage(content=f"[Etape 1] {output}")],
        }

    async def node_risks(self, state: AgentState) -> dict:
        step = "risks"
        ctx_struct = self._find_step(state, "context")
        meds = ctx_struct.get("medications", [])
        query = "risks, abnormal results, negative lab trends, missed follow-up"
        ctx = self._memory_context(query)
        user = (
            f"## Existing memory context\n{ctx.text or '(empty - rely on tools)'}\n\n"
            f"Known medications from step 1: "
            f"{', '.join(meds) if meds else 'none documented'}.\n\n"
            "Investigate risks: call analyze_lab_trends for lab trends, "
            "check_drug_interactions with the medication list, rag_search_documents "
            "for follow-up gaps, and web_search_tool for guidelines if useful. Then "
            "enumerate clinically significant risks, each with evidence and source."
            + self._TOOL_MENU
        )
        parsed, tool_records, token_records = await llm.run_agentic_step(
            step,
            prompts.STEP_RISKS_SYSTEM,
            user,
            self._tools(
                [
                    "rag_search_documents",
                    "analyze_lab_trends",
                    "check_drug_interactions",
                    "web_search_tool",
                ]
            ),
            RiskResult,
        )
        output = self._format_risks(parsed)
        return {
            "current_step": step,
            "step_results": [
                {
                    "step_id": step,
                    "title": "2. Detecter les risques",
                    "structured": parsed.model_dump(),
                    "output": output,
                    "context_used": ctx.to_dict(),
                }
            ],
            "tool_calls": tool_records,
            "token_ledger": token_records,
            "pending_memory": {"step": step, "text": output},
            "messages": [AIMessage(content=f"[Etape 2] {output}")],
        }

    async def node_prioritize(self, state: AgentState) -> dict:
        step = "prioritize"
        risks = self._find_step(state, "risks").get("risks", [])
        risk_text = "\n".join(
            f"- [{r.get('severity')}] {r.get('description')} (source: {r.get('source')})"
            for r in risks
        ) or "No risks recorded."
        user = (
            f"Identified risks:\n{risk_text}\n\n"
            "Classify each finding as urgent, can-wait, or needs-clarification. "
            "If a finding is ambiguous, you may call rag_search_documents to verify "
            "before deciding." + self._TOOL_MENU
        )
        parsed, tool_records, token_records = await llm.run_agentic_step(
            step,
            prompts.STEP_PRIORITIZE_SYSTEM,
            user,
            self._tools(["rag_search_documents"]),
            PriorityResult,
            max_iters=2,
        )
        output = self._format_priority(parsed)
        return {
            "current_step": step,
            "step_results": [
                {
                    "step_id": step,
                    "title": "3. Prioriser",
                    "structured": parsed.model_dump(),
                    "output": output,
                }
            ],
            "tool_calls": tool_records,
            "token_ledger": token_records,
            "pending_memory": {"step": step, "text": output},
            "messages": [AIMessage(content=f"[Etape 3] {output}")],
        }

    async def node_actions(self, state: AgentState) -> dict:
        step = "actions"
        priorities = self._find_step(state, "prioritize")
        risks = self._find_step(state, "risks").get("risks", [])
        user = (
            f"Prioritized findings: {json.dumps(priorities, ensure_ascii=False)}\n\n"
            f"Risks: {json.dumps(risks, ensure_ascii=False)}\n\n"
            "Propose physician questions, behavior changes, exams to request, and "
            "follow-up reminders. You may call web_search_tool for current guidelines "
            "or rag_search_documents to ground a recommendation." + self._TOOL_MENU
        )
        parsed, tool_records, token_records = await llm.run_agentic_step(
            step,
            prompts.STEP_ACTIONS_SYSTEM,
            user,
            self._tools(["rag_search_documents", "web_search_tool"]),
            ActionResult,
        )
        output = self._format_actions(parsed)
        return {
            "current_step": step,
            "step_results": [
                {
                    "step_id": step,
                    "title": "4. Produire des actions",
                    "structured": parsed.model_dump(),
                    "output": output,
                }
            ],
            "tool_calls": tool_records,
            "token_ledger": token_records,
            "pending_memory": {"step": step, "text": output},
            "messages": [AIMessage(content=f"[Etape 4] {output}")],
        }

    async def node_synthesis(self, state: AgentState) -> dict:
        step = "synthesis"
        context = self._find_step(state, "context")
        risks = self._find_step(state, "risks")
        priorities = self._find_step(state, "prioritize")
        actions = self._find_step(state, "actions")
        user = (
            f"Language for the summary: {self.language}.\n\n"
            f"Context: {json.dumps(context, ensure_ascii=False)}\n"
            f"Risks: {json.dumps(risks, ensure_ascii=False)}\n"
            f"Priorities: {json.dumps(priorities, ensure_ascii=False)}\n"
            f"Actions: {json.dumps(actions, ensure_ascii=False)}\n\n"
            "Write a simplified, patient-friendly summary."
        )
        # Synthesis is a pure summarization step (no further tool digging).
        parsed, tool_records, token_records = await llm.run_agentic_step(
            step,
            prompts.STEP_SYNTHESIS_SYSTEM,
            user,
            [],
            SynthesisResult,
        )
        output = parsed.patient_summary
        return {
            "current_step": step,
            "step_results": [
                {
                    "step_id": step,
                    "title": "5. Synthese patient",
                    "structured": parsed.model_dump(),
                    "output": output,
                }
            ],
            "tool_calls": tool_records,
            "token_ledger": token_records,
            "final_summary": output,
            "pending_memory": {"step": step, "text": output},
            "messages": [AIMessage(content=f"[Etape 5] {output}")],
        }

    async def node_memory_eval(self, state: AgentState) -> dict:
        pending = state.get("pending_memory") or {}
        if not pending:
            return {}
        step = pending.get("step", "?")
        text = pending.get("text", "")

        # Durable knowledge is summarized into long-term memory via the MCP tool.
        tool_calls = []
        summary = None
        if self.memory.should_store_long_term(step, text):
            summary, _, rec = await self.call_tool(
                "memory", "summarize_history", {"text": text, "focus": step}
            )
            tool_calls.append(rec)

        decision = self.memory.evaluate_and_store(
            step, text, precomputed_summary=summary
        )
        return {
            "memory_decisions": [decision],
            "memory_snapshot": self.memory.snapshot(),
            "tool_calls": tool_calls,
            "pending_memory": {},
        }

    # ------------------------------------------------------------- formatting
    @staticmethod
    def _format_context(c: ContextResult) -> str:
        parts = [c.longitudinal_portrait]
        if c.chronic_conditions:
            parts.append("Conditions chroniques: " + ", ".join(c.chronic_conditions))
        if c.medications:
            parts.append("Medicaments: " + ", ".join(c.medications))
        if c.key_documents:
            parts.append("Documents cles: " + ", ".join(c.key_documents))
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _format_risks(r: RiskResult) -> str:
        if not r.risks:
            return "Aucun risque significatif identifie dans les documents fournis."
        return "\n".join(
            f"- [{item.severity}] {item.description} | preuve: {item.evidence} "
            f"(source: {item.source})"
            for item in r.risks
        )

    @staticmethod
    def _format_priority(p: PriorityResult) -> str:
        return (
            "Urgent: " + ("; ".join(p.urgent) or "-") + "\n"
            "Peut attendre: " + ("; ".join(p.can_wait) or "-") + "\n"
            "A clarifier: " + ("; ".join(p.needs_clarification) or "-")
        )

    @staticmethod
    def _format_actions(a: ActionResult) -> str:
        return (
            "Questions au medecin: " + ("; ".join(a.physician_questions) or "-") + "\n"
            "Changements de comportement: " + ("; ".join(a.behavior_changes) or "-") + "\n"
            "Examens a demander: " + ("; ".join(a.exams_to_request) or "-") + "\n"
            "Rappels de suivi: " + ("; ".join(a.followup_reminders) or "-")
        )

    # ----------------------------------------------------------------- greet
    async def greet(self) -> tuple[str, dict]:
        user = (
            f"Language: {self.language}. The user is preparing to review a patient "
            "chart. Greet them and explain the 5-step process."
        )
        text, tok = await llm.ainvoke_text(
            "welcome", prompts.WELCOME_SYSTEM, user, temperature=0.5
        )
        return text, tok

    # ------------------------------------------------------------- follow-up
    async def answer_followup(self, question: str) -> dict:
        step = "followup"
        ctx = self._memory_context(question)
        user = (
            f"Question: {question}\n\n"
            f"## Existing memory context\n{ctx.text or '(empty - rely on tools)'}\n\n"
            "Search the documents (and the web if relevant) to answer precisely, "
            "then cite sources." + self._TOOL_MENU
        )
        answer, tool_records, token_records = await llm.run_agentic_text(
            step,
            prompts.FOLLOWUP_SYSTEM,
            user,
            self._tools(["rag_search_documents", "web_search_tool"]),
        )
        self.memory.add_turn("user", question)
        self.memory.add_turn("assistant", answer)
        return {
            "answer": answer,
            "token_records": token_records,
            "tool_calls": tool_records,
            "memory_snapshot": self.memory.snapshot(),
        }

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("context", self.node_context)
        g.add_node("risks", self.node_risks)
        g.add_node("prioritize", self.node_prioritize)
        g.add_node("actions", self.node_actions)
        g.add_node("synthesis", self.node_synthesis)
        # One memory-evaluation implementation, registered after each step.
        g.add_node("mem_context", self.node_memory_eval)
        g.add_node("mem_risks", self.node_memory_eval)
        g.add_node("mem_prioritize", self.node_memory_eval)
        g.add_node("mem_actions", self.node_memory_eval)
        g.add_node("mem_synthesis", self.node_memory_eval)

        g.add_edge(START, "context")
        g.add_edge("context", "mem_context")
        g.add_edge("mem_context", "risks")
        g.add_edge("risks", "mem_risks")
        g.add_edge("mem_risks", "prioritize")
        g.add_edge("prioritize", "mem_prioritize")
        g.add_edge("mem_prioritize", "actions")
        g.add_edge("actions", "mem_actions")
        g.add_edge("mem_actions", "synthesis")
        g.add_edge("synthesis", "mem_synthesis")
        g.add_edge("mem_synthesis", END)
        return g.compile()

    async def run(self, user_request: str) -> dict:
        initial: AgentState = {
            "patient_id": self.patient_id,
            "language": self.language,
            "user_request": user_request,
            "messages": [],
            "step_results": [],
            "tool_calls": [],
            "token_ledger": [],
            "memory_decisions": [],
            "memory_snapshot": self.memory.snapshot(),
        }
        return await self.graph.ainvoke(initial)

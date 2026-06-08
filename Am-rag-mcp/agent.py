"""
Agent orchestrateur clinique  pour Streamlit.
Expose une classe ClinicalAgent avec des serveurs MCP persistants.
"""
import asyncio
import json
import sys
import warnings
from pathlib import Path
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

warnings.filterwarnings("ignore")

PHASE1_SRC = Path(__file__).parent.parent / "Am-rag" / "src"
sys.path.insert(0, str(PHASE1_SRC))

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_CHAT_DEPLOYMENT,
)
from openai import AzureOpenAI as AzureClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


TOOLS_DIR = Path(__file__).parent / "tools"
MCP_SERVERS = {
    "rag": TOOLS_DIR / "rag_tool.py",
    "timeline": TOOLS_DIR / "timeline_tool.py",
    "summary": TOOLS_DIR / "summary_tool.py",
    "risk": TOOLS_DIR / "risk_tool.py",
}

SYSTEM_PROMPT = """You are a clinical preparation assistant helping a physician prepare for a patient consultation.

You have access to several tools that provide structured information about the patient:
- search_clinical_documents: semantic search in clinical notes (best for specific factual questions)
- get_patient_timeline: chronological events with date filters (best for "when" questions, evolutions)
- get_patient_summary: pre-computed structured patient summary (best for overviews)
- get_clinical_risks: pre-computed risk analysis with priorities (best for "what to worry about" questions)

Strategy:
- For broad questions, start with summary and risks.
- For specific factual questions, prefer search_clinical_documents.
- For temporal questions, use timeline.
- Combine multiple tools when needed.
- Cite sources when available (file name, date).
- Be concise and structured. Use medical terminology appropriate for a physician.
- Answer in the same language as the user's question (French or English).
"""


@dataclass
class TraceStep:
    """Une etape de raisonnement pour l'affichage UI."""
    kind: str  # "tool_call" | "tool_result" | "final_answer"
    tool_name: str = ""
    arguments: dict = field(default_factory=dict)
    content: str = ""


class MCPClientWrapper:
    def __init__(self, name: str, script_path: Path):
        self.name = name
        self.script_path = script_path
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self.tools_info: list = []

    async def connect(self):
        params = StdioServerParameters(command="python", args=[str(self.script_path)])
        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(params))
        read, write = stdio_transport
        self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        result = await self.session.list_tools()
        self.tools_info = result.tools
        return self.tools_info

    async def call(self, tool_name: str, arguments: dict) -> str:
        result = await self.session.call_tool(tool_name, arguments=arguments)
        texts = [c.text for c in result.content if hasattr(c, "text")]
        return "\n".join(texts)

    async def close(self):
        try:
            await asyncio.wait_for(self._exit_stack.aclose(), timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass


def _mcp_to_openai(mcp_tool, server_name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": f"{server_name}__{mcp_tool.name}",
            "description": (mcp_tool.description or "").strip(),
            "parameters": mcp_tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


class ClinicalAgent:
    """Agent persistant qui garde les serveurs MCP vivants entre les questions."""

    def __init__(self):
        self.clients: dict[str, MCPClientWrapper] = {}
        self.openai_tools: list = []
        self.tool_routing: dict[str, tuple[MCPClientWrapper, str]] = {}
        self.azure_client = AzureClient(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )

    async def start(self, progress_callback=None):
        """Demarre tous les serveurs MCP. progress_callback(name) appele a chaque demarrage."""
        for name, path in MCP_SERVERS.items():
            if progress_callback:
                progress_callback(name)
            client = MCPClientWrapper(name, path)
            tools_info = await client.connect()
            self.clients[name] = client
            for t in tools_info:
                schema = _mcp_to_openai(t, name)
                self.openai_tools.append(schema)
                self.tool_routing[schema["function"]["name"]] = (client, t.name)

    async def stop(self):
        for client in self.clients.values():
            await client.close()

    async def ask(self, question: str, max_iterations: int = 6) -> tuple[str, list[TraceStep]]:
        """Pose une question a l'agent. Retourne (reponse_finale, trace)."""
        trace: list[TraceStep] = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for _ in range(max_iterations):
            response = self.azure_client.chat.completions.create(
                model=AZURE_CHAT_DEPLOYMENT,
                messages=messages,
                tools=self.openai_tools,
                tool_choice="auto",
                temperature=0.1,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                trace.append(TraceStep(kind="final_answer", content=msg.content or ""))
                return msg.content or "", trace

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                fname = tc.function.name
                fargs = json.loads(tc.function.arguments or "{}")
                trace.append(TraceStep(kind="tool_call", tool_name=fname, arguments=fargs))

                if fname not in self.tool_routing:
                    result = f"ERROR: unknown tool {fname}"
                else:
                    client, mcp_name = self.tool_routing[fname]
                    try:
                        result = await client.call(mcp_name, fargs)
                    except Exception as e:
                        result = f"ERROR calling {fname}: {e}"

                result_short = result[:8000] if len(result) > 8000 else result
                trace.append(TraceStep(kind="tool_result", tool_name=fname, content=result_short))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_short,
                })

        return "Limite d'iterations atteinte sans reponse finale.", trace


# Mode CLI pour tester rapidement
async def _cli():
    agent = ClinicalAgent()
    print("Demarrage des serveurs MCP...")
    await agent.start(progress_callback=lambda n: print(f"  [{n}]"))
    print(f"Pret. {len(agent.openai_tools)} outils disponibles.\n")
    try:
        while True:
            try:
                q = input("Question > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit", ""):
                break
            answer, trace = await agent.ask(q)
            print("\nReasoning:")
            for step in trace:
                if step.kind == "tool_call":
                    print(f"  -> {step.tool_name}({step.arguments})")
                elif step.kind == "tool_result":
                    print(f"     <- {len(step.content)} chars")
            print(f"\nReponse:\n{answer}\n")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(_cli())
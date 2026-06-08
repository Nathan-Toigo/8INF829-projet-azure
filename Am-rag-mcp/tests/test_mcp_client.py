"""Test du serveur MCP via un client direct (sans inspector)."""
import asyncio
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).parent.parent / "tools" / "rag_tool.py"


async def main():
    params = StdioServerParameters(command="python", args=[str(SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Lister les outils exposes
            tools = await session.list_tools()
            print("Outils exposes par le serveur:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description.strip()[:80]}...")

            # 2. Appeler l'outil
            print("\nAppel de search_clinical_documents...")
            result = await session.call_tool(
                "search_clinical_documents",
                arguments={"question": "What medications is the patient taking?"},
            )
            for content in result.content:
                print(content.text)


if __name__ == "__main__":
    asyncio.run(main())
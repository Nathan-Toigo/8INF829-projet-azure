"""Test du serveur MCP Timeline via un client direct."""
import asyncio
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).parent.parent / "tools" / "timeline_tool.py"


async def main():
    params = StdioServerParameters(command="python", args=[str(SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Outils exposes par le serveur Timeline:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description.strip()[:80]}...")

            print("\nAppel: get_patient_timeline(start_date=2021-10-01, end_date=2021-12-31)")
            result = await session.call_tool(
                "get_patient_timeline",
                arguments={"start_date": "2021-10-01", "end_date": "2021-12-31"},
            )
            for content in result.content:
                print(content.text[:2000])


if __name__ == "__main__":
    asyncio.run(main())
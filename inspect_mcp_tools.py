import os
import json
import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def main():
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_PERSONAL_ACCESS_TOKEN is not set")

    params = StdioServerParameters(
        command="docker",
        args=[
            "run", "-i", "--rm",
            "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
            "-e", "GITHUB_TOOLSETS=issues,repos",
            "-e", "GITHUB_DYNAMIC_TOOLSETS=1",
            "-e", "GITHUB_READ_ONLY=1",
            "ghcr.io/github/github-mcp-server"
        ],
        env={
            "GITHUB_PERSONAL_ACCESS_TOKEN": token
        }
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

            for tool in tools.tools:
                if tool.name in ["list_issues", "issue_read", "search_issues"]:
                    print("\n" + "=" * 80)
                    print("TOOL:", tool.name)
                    print("=" * 80)
                    print(json.dumps(getattr(tool, "inputSchema", {}), indent=2))

asyncio.run(main())
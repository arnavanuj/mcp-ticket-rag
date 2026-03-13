import os
import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def main():
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")

    print("Token present in this terminal session:", bool(token))

    if not token:
        raise RuntimeError("GITHUB_PERSONAL_ACCESS_TOKEN is not set in this PowerShell session")

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
            print("Initializing MCP session...")
            await session.initialize()
            print("MCP session initialized.")

            tools = await session.list_tools()
            print("\nDiscovered MCP tools:\n")

            for tool in tools.tools:
                print("-", tool.name)

asyncio.run(main())
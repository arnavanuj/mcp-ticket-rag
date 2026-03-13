"""GitHub MCP client with exact schema-compliant argument mapping."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.mcp.adapters import adapt_comment, adapt_issue


@dataclass(slots=True)
class GitHubMCPClient:
    """MCP-only GitHub access layer for issues/comments."""

    settings: Settings
    server_name: str = "github"
    logger: Any = field(init=False)
    mcp_connected: bool = field(init=False, default=False)
    mcp_transport: str = field(init=False, default="")
    discovered_tools: list[str] = field(init=False, default_factory=list)
    tool_schemas: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
    used_tools: list[str] = field(init=False, default_factory=list)
    rest_fallback_used: bool = field(init=False, default=False)
    zero_result_mcp_call: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.logger = get_logger("app.mcp.github_client")
        self._initialize_mcp()

    def _initialize_mcp(self) -> None:
        tools, schemas = asyncio.run(self._discover_tools_async())
        self.discovered_tools = tools
        self.tool_schemas = schemas
        self.mcp_connected = bool(tools)
        self.logger.info("MCP_CONNECTED | server=%s | transport=%s", self.server_name, self.mcp_transport)
        self.logger.info("MCP_TOOLS_DISCOVERED | tools=%s", tools)

    async def _discover_tools_async(self) -> tuple[list[str], dict[str, dict[str, Any]]]:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        if self.settings.github_mcp_transport != "stdio":
            raise RuntimeError("This run requires local stdio MCP transport")
        if not self.settings.github_token:
            raise RuntimeError(
                "GITHUB_PERSONAL_ACCESS_TOKEN is required for MCP-first GitHub server execution"
            )

        self.mcp_transport = "stdio"
        args = shlex.split(self.settings.github_mcp_server_args)
        env = {}
        if self.settings.github_token:
            env["GITHUB_PERSONAL_ACCESS_TOKEN"] = self.settings.github_token

        params = StdioServerParameters(command=self.settings.github_mcp_server_command, args=args, env=env)

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tool_result = await session.list_tools()
                tools: list[str] = []
                schemas: dict[str, dict[str, Any]] = {}
                for tool in tool_result.tools:
                    tools.append(tool.name)
                    schemas[tool.name] = getattr(tool, "inputSchema", None) or {}
                return tools, schemas

    def _require_tool(self, name: str) -> None:
        if name not in self.discovered_tools:
            raise RuntimeError(f"Required MCP tool not discovered: {name}")

    def _log_tool_used(self, tool: str) -> None:
        if tool not in self.used_tools:
            self.used_tools.append(tool)

    async def _call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        self.logger.info(
            "MCP_TOOL_SELECTED | tool=%s | schema=%s",
            tool_name,
            self.tool_schemas.get(tool_name, {}),
        )
        self.logger.info(
            "MCP_CALL | server=%s | tool=%s | params=%s | transport=%s",
            self.server_name,
            tool_name,
            arguments,
            self.mcp_transport,
        )

        args = shlex.split(self.settings.github_mcp_server_args)
        env = {}
        if self.settings.github_token:
            env["GITHUB_PERSONAL_ACCESS_TOKEN"] = self.settings.github_token
        params = StdioServerParameters(command=self.settings.github_mcp_server_command, args=args, env=env)

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                self._log_tool_used(tool_name)
                return self._extract_payload(result)

    def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return asyncio.run(self._call_tool_async(tool_name, arguments))

    @staticmethod
    def _extract_payload(result: Any) -> Any:
        content = getattr(result, "content", None)
        if not isinstance(content, list):
            return result

        values: list[Any] = []
        for item in content:
            item_text = getattr(item, "text", None)
            if item_text:
                try:
                    values.append(json.loads(item_text))
                    continue
                except Exception:
                    values.append(item_text)
                    continue
            item_data = getattr(item, "data", None)
            if item_data is not None:
                values.append(item_data)

        if len(values) == 1:
            return values[0]
        return values

    @staticmethod
    def _truncate_for_log(value: Any, max_chars: int = 5000) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str)
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}...<truncated {len(text) - max_chars} chars>"

    @staticmethod
    def _normalize_items(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            out: list[dict[str, Any]] = []
            for part in payload:
                if isinstance(part, dict) and isinstance(part.get("items"), list):
                    out.extend(x for x in part["items"] if isinstance(x, dict))
                elif isinstance(part, dict):
                    out.append(part)
            return out
        if isinstance(payload, dict):
            if isinstance(payload.get("items"), list):
                return [x for x in payload["items"] if isinstance(x, dict)]
            if isinstance(payload.get("issues"), list):
                return [x for x in payload["issues"] if isinstance(x, dict)]
            return [payload]
        return []

    @staticmethod
    def _derive_issue_number(item: dict[str, Any]) -> int | None:
        candidates = [
            item.get("number"),
            item.get("issue_number"),
            item.get("id"),
        ]
        for value in candidates:
            try:
                if value is not None:
                    return int(value)
            except Exception:
                continue

        for key in ("url", "html_url"):
            url = item.get(key)
            if isinstance(url, str):
                match = re.search(r"/issues/(\\d+)", url)
                if match:
                    try:
                        return int(match.group(1))
                    except Exception:
                        pass
        return None

    @staticmethod
    def _coerce_for_adapt_issue(item: dict[str, Any], issue_number: int) -> dict[str, Any]:
        if "number" in item:
            return item
        out = dict(item)
        out["number"] = issue_number
        if "title" not in out and "issue_title" in out:
            out["title"] = out.get("issue_title")
        if "body" not in out and "issue_body" in out:
            out["body"] = out.get("issue_body")
        if "html_url" not in out and "url" in out:
            out["html_url"] = out.get("url")
        return out

    def list_closed_issues(self, owner: str, repo: str, per_page: int = 20, page: int = 1) -> list[dict[str, Any]]:
        self._require_tool("list_issues")
        self._require_tool("issue_read")

        payload = self._call_tool(
            "list_issues",
            {
                "owner": owner,
                "repo": repo,
                "state": "CLOSED",
                "orderBy": "UPDATED_AT",
                "direction": "DESC",
                "perPage": per_page,
            },
        )
        self.logger.info("MCP_RAW_PAYLOAD | tool=list_issues | payload_type=%s", type(payload).__name__)
        self.logger.info("MCP_RAW_PAYLOAD | tool=list_issues | payload=%s", self._truncate_for_log(payload))

        raw_items = self._normalize_items(payload)
        self.logger.info("MCP_NORMALIZED_ITEMS | tool=list_issues | count=%s", len(raw_items))
        self.logger.info("MCP_NORMALIZED_ITEMS | tool=list_issues | items=%s", self._truncate_for_log(raw_items))
        for idx, item in enumerate(raw_items[:3], start=1):
            keys = sorted(item.keys()) if isinstance(item, dict) else []
            self.logger.info("MCP_ITEM_KEYS | tool=list_issues | index=%s | keys=%s", idx, keys)
            if isinstance(item, dict):
                self.logger.info(
                    "MCP_ITEM_ID_FIELDS | index=%s | number=%s | issue_number=%s | id=%s | node_id=%s | url=%s | title=%s",
                    idx,
                    item.get("number"),
                    item.get("issue_number"),
                    item.get("id"),
                    item.get("node_id"),
                    item.get("url"),
                    item.get("title"),
                )

        if len(raw_items) == 0:
            self.zero_result_mcp_call = True
            self.logger.error("MCP_RESULT_ZERO | tool=list_issues | raw_payload=%s", payload)
            raise RuntimeError("MCP list_issues returned zero items; stopping as requested")

        detailed: list[dict[str, Any]] = []
        for item in raw_items[:per_page]:
            issue_number = self._derive_issue_number(item)
            if not issue_number:
                continue
            issue_payload = self._call_tool(
                "issue_read",
                {
                    "owner": owner,
                    "repo": repo,
                    "issue_number": issue_number,
                    "method": "get",
                },
            )
            issue_items = self._normalize_items(issue_payload)
            if issue_items:
                shaped = self._coerce_for_adapt_issue(issue_items[0], issue_number=issue_number)
                detailed.append(adapt_issue(shaped))

        self.logger.info("MCP_RESULT | tool=list_issues+issue_read | count=%s", len(detailed))
        return detailed

    def get_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        """Fetch a single issue through MCP issue_read(get)."""

        self._require_tool("issue_read")
        payload = self._call_tool(
            "issue_read",
            {
                "owner": owner,
                "repo": repo,
                "issue_number": issue_number,
                "method": "get",
            },
        )
        items = self._normalize_items(payload)
        if not items:
            raise ValueError(f"Issue {issue_number} not found via MCP")
        shaped = self._coerce_for_adapt_issue(items[0], issue_number=issue_number)
        issue = adapt_issue(shaped)
        self.logger.info(
            "MCP_RESULT | tool=issue_read(get) | issue_number=%s | title=%s",
            issue_number,
            issue.get("issue_title"),
        )
        return issue

    def get_issue_raw(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        """Fetch single issue raw payload for dynamic field introspection."""

        self._require_tool("issue_read")
        payload = self._call_tool(
            "issue_read",
            {
                "owner": owner,
                "repo": repo,
                "issue_number": issue_number,
                "method": "get",
            },
        )
        items = self._normalize_items(payload)
        if not items:
            raise ValueError(f"Issue {issue_number} not found via MCP")
        item = dict(items[0])
        if "number" not in item:
            item["number"] = issue_number
        return item

    def get_issue_comments_raw(self, owner: str, repo: str, issue_number: int) -> list[dict[str, Any]]:
        """Fetch raw comments payload for dynamic field introspection."""

        self._require_tool("issue_read")
        comments: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self._call_tool(
                "issue_read",
                {
                    "owner": owner,
                    "repo": repo,
                    "issue_number": issue_number,
                    "method": "get_comments",
                    "perPage": 100,
                    "page": page,
                },
            )
            items = self._normalize_items(payload)
            if not items:
                break
            raw_page = [dict(item) for item in items if isinstance(item, dict)]
            comments.extend(raw_page)
            if len(raw_page) < 100:
                break
            page += 1
        return comments

    def list_issue_comments(self, owner: str, repo: str, issue_number: int) -> list[dict[str, Any]]:
        self._require_tool("issue_read")

        comments: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self._call_tool(
                "issue_read",
                {
                    "owner": owner,
                    "repo": repo,
                    "issue_number": issue_number,
                    "method": "get_comments",
                    "perPage": 100,
                    "page": page,
                },
            )
            items = self._normalize_items(payload)
            if not items:
                break

            page_comments = [adapt_comment(item, issue_number) for item in items if isinstance(item, dict)]
            comments.extend(page_comments)
            if len(page_comments) < 100:
                break
            page += 1

        self.logger.info(
            "MCP_RESULT | tool=issue_read(get_comments) | issue_number=%s | count=%s",
            issue_number,
            len(comments),
        )
        return comments

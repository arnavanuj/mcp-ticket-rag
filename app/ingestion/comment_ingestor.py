"""Comment ingestion helpers."""

from __future__ import annotations

from app.mcp.github_client import GitHubMCPClient


def ingest_comments(client: GitHubMCPClient, owner: str, repo: str, issue_number: int) -> list[dict]:
    """Fetch comments for a single issue."""

    return client.list_issue_comments(owner=owner, repo=repo, issue_number=issue_number)

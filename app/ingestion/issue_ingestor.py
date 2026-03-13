"""Issue ingestion helpers."""

from __future__ import annotations

from app.mcp.github_client import GitHubMCPClient


def ingest_issues(client: GitHubMCPClient, owner: str, repo: str, max_issues: int) -> list[dict]:
    """Fetch latest closed issues with conservative pagination."""

    issues: list[dict] = []
    page = 1
    per_page = min(max_issues, 20)

    while len(issues) < max_issues:
        page_items = client.list_closed_issues(owner=owner, repo=repo, per_page=per_page, page=page)
        if not page_items:
            break
        remaining = max_issues - len(issues)
        issues.extend(page_items[:remaining])
        page += 1

    return issues

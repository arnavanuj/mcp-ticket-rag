"""Ticket service for reading ingested issue data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import Settings


def get_ticket(settings: Settings, issue_number: int) -> dict[str, Any]:
    """Return an ingested issue and related comments from raw files."""

    issues_path = settings.raw_dir / "issues.json"
    comments_path = settings.raw_dir / "comments.json"

    if not issues_path.exists():
        raise FileNotFoundError("Ingestion not run yet; issues.json missing")

    issues = json.loads(issues_path.read_text(encoding="utf-8"))
    issue = next((item for item in issues if item.get("issue_number") == issue_number), None)
    if issue is None:
        raise ValueError(f"Issue {issue_number} not found in ingested dataset")

    comments: dict[str, list[dict[str, Any]]] = {}
    if comments_path.exists():
        comments = json.loads(comments_path.read_text(encoding="utf-8"))

    assignees = issue.get("assignees") or ["unassigned"]
    assignee_names = issue.get("assignee_names") or []
    return {
        "issue_number": issue_number,
        "issue_title": issue.get("issue_title", ""),
        "issue_url": issue.get("url", ""),
        "closed_at": issue.get("closed_at"),
        "assignees": assignees,
        "assignee_names": assignee_names,
        "comments": comments.get(str(issue_number), comments.get(issue_number, [])),
    }

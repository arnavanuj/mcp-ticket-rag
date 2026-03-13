"""Adapters for translating GitHub payloads into internal shapes."""
from __future__ import annotations

from typing import Any


def _normalize_labels(labels: list[Any] | None) -> list[str]:
    if not labels:
        return []

    out: list[str] = []
    for label in labels:
        if isinstance(label, dict):
            out.append(str(label.get("name", "")))
        elif isinstance(label, str):
            out.append(label)
        else:
            out.append(str(label))
    return [x for x in out if x]


def adapt_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """Normalize selected GitHub issue fields used by the pipeline."""
    assignees = [user for user in issue.get("assignees", []) if isinstance(user, dict)]
    assignee_usernames = [str(user.get("login", "")).strip() for user in assignees if user.get("login")]
    assignee_names = [str(user.get("name", "")).strip() for user in assignees if user.get("name")]

    return {
        "issue_number": issue.get("number"),
        "issue_title": issue.get("title", ""),
        "issue_body": issue.get("body") or "",
        "labels": _normalize_labels(issue.get("labels")),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
        "assignees": assignee_usernames,
        "assignee_usernames": assignee_usernames,
        "assignee_names": assignee_names,
        "author": (issue.get("user") or {}).get("login", "unknown") if isinstance(issue.get("user"), dict) else "unknown",
        "url": issue.get("html_url") or issue.get("url", ""),
        "comments_count": issue.get("comments", 0),
    }


def adapt_comment(comment: dict[str, Any], issue_number: int) -> dict[str, Any]:
    """Normalize selected GitHub comment fields used by the pipeline."""
    user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
    author_login = user.get("login", "unknown")
    author_name = user.get("name") or ""

    return {
        "source_type": "comment",
        "issue_number": issue_number,
        "comment_id": comment.get("id"),
        "author": author_login,
        "author_login": author_login,
        "author_name": author_name,
        "created_at": comment.get("created_at"),
        "updated_at": comment.get("updated_at"),
        "url": comment.get("html_url") or comment.get("url", ""),
        "raw_text": comment.get("body") or "",
    }

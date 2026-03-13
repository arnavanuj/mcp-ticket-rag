"""Normalize issues/comments/images into a unified document schema."""

from __future__ import annotations

from typing import Any


def normalize_issue(owner: str, repo: str, issue: dict[str, Any]) -> dict[str, Any]:
    assignees = issue.get("assignee_usernames") or issue.get("assignees") or []
    assignee_names = issue.get("assignee_names") or []
    assignee_str = ", ".join(assignees) if assignees else "unassigned"
    assignee_name_str = ", ".join(assignee_names) if assignee_names else "not clearly stated"
    text = (
        f"Issue #{issue['issue_number']}: {issue.get('issue_title', '')}\n"
        f"Assignees (usernames): {assignee_str}\n"
        f"Assignee names: {assignee_name_str}\n"
        f"Labels: {', '.join(issue.get('labels', []))}\n"
        f"Body:\n{issue.get('issue_body', '')}"
    )

    return {
        "source_type": "issue",
        "repo": repo,
        "owner": owner,
        "issue_number": issue.get("issue_number"),
        "comment_id": None,
        "title": issue.get("issue_title"),
        "url": issue.get("url"),
        "author": issue.get("author", "unknown"),
        "assignees": assignees,
        "assignee_names": assignee_names,
        "comment_author": "",
        "comment_author_name": "",
        "created_at": issue.get("created_at"),
        "closed_at": issue.get("closed_at"),
        "labels": issue.get("labels", []),
        "raw_text": text,
        "image_ocr_text": "",
        "image_analysis_text": "",
        "source_path": "",
    }


def normalize_comment(owner: str, repo: str, issue_number: int, comment: dict[str, Any], title: str) -> dict[str, Any]:
    author_login = comment.get("author_login") or comment.get("author") or "unknown"
    author_name = comment.get("author_name") or ""
    author_name_str = author_name if author_name else "not clearly stated"
    raw_text = (
        f"Comment author username: {author_login}\n"
        f"Comment author name: {author_name_str}\n"
        f"Comment body:\n{comment.get('raw_text', '')}"
    )

    return {
        "source_type": "comment",
        "repo": repo,
        "owner": owner,
        "issue_number": issue_number,
        "comment_id": comment.get("comment_id"),
        "title": title,
        "url": comment.get("url"),
        "author": author_login,
        "assignees": [],
        "assignee_names": [],
        "comment_author": author_login,
        "comment_author_name": author_name,
        "created_at": comment.get("created_at"),
        "closed_at": None,
        "labels": [],
        "raw_text": raw_text,
        "image_ocr_text": "",
        "image_analysis_text": "",
        "source_path": "",
    }


def normalize_image_ocr(
    owner: str,
    repo: str,
    issue_number: int,
    title: str,
    image_url: str,
    ocr_text: str,
    source_path: str,
) -> dict[str, Any]:
    return {
        "source_type": "image_ocr",
        "repo": repo,
        "owner": owner,
        "issue_number": issue_number,
        "comment_id": None,
        "title": title,
        "url": image_url,
        "author": "system",
        "assignees": [],
        "assignee_names": [],
        "comment_author": "",
        "comment_author_name": "",
        "created_at": None,
        "closed_at": None,
        "labels": [],
        "raw_text": "",
        "image_ocr_text": ocr_text,
        "image_analysis_text": "",
        "source_path": source_path,
    }


def normalize_image_analysis(
    owner: str,
    repo: str,
    issue_number: int,
    title: str,
    image_url: str,
    analysis_text: str,
    source_path: str,
) -> dict[str, Any]:
    return {
        "source_type": "image_analysis",
        "repo": repo,
        "owner": owner,
        "issue_number": issue_number,
        "comment_id": None,
        "title": title,
        "url": image_url,
        "author": "system",
        "assignees": [],
        "assignee_names": [],
        "comment_author": "",
        "comment_author_name": "",
        "created_at": None,
        "closed_at": None,
        "labels": [],
        "raw_text": "",
        "image_ocr_text": "",
        "image_analysis_text": analysis_text,
        "source_path": source_path,
    }

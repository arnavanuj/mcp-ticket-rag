"""Citation and evidence data models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """Evidence citation returned with every grounded answer."""

    source_type: str = Field(description="issue|comment|image_ocr|repo_file")
    issue_number: int | None = None
    comment_id: int | None = None
    title: str | None = None
    url: str
    assignees: str | None = None
    assignee_names: str | None = None
    comment_author: str | None = None
    comment_author_name: str | None = None
    snippet: str

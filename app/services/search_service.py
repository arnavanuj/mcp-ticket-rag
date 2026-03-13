"""Search service over processed normalized documents."""

from __future__ import annotations

import json

from app.core.config import Settings


def search_tickets(settings: Settings, query: str, limit: int = 20) -> list[dict]:
    """Perform lightweight keyword search over normalized docs."""

    processed_path = settings.processed_dir / "normalized_docs.json"
    if not processed_path.exists():
        return []

    docs = json.loads(processed_path.read_text(encoding="utf-8"))
    q = query.strip().lower()
    results: list[dict] = []

    for doc in docs:
        hay = "\n".join(
            [
                str(doc.get("title", "")),
                str(doc.get("raw_text", "")),
                str(doc.get("image_ocr_text", "")),
                str(doc.get("image_analysis_text", "")),
                str(doc.get("assignees", "")),
                str(doc.get("assignee_names", "")),
                str(doc.get("comment_author", "")),
                str(doc.get("comment_author_name", "")),
            ]
        ).lower()
        if q and q in hay:
            results.append(
                {
                    "source_type": doc.get("source_type"),
                    "issue_number": doc.get("issue_number"),
                    "comment_id": doc.get("comment_id"),
                    "title": doc.get("title"),
                    "url": doc.get("url"),
                }
            )
        if len(results) >= limit:
            break

    return results

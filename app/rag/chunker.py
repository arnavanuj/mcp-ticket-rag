"""Small text chunker that preserves metadata per chunk."""

from __future__ import annotations

import hashlib
from typing import Any


def chunk_documents(docs: list[dict[str, Any]], chunk_size: int = 700, overlap: int = 120) -> list[dict[str, Any]]:
    """Chunk each document into small overlapping windows."""

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[dict[str, Any]] = []
    stride = chunk_size - overlap

    for doc in docs:
        text_parts = [
            (doc.get("raw_text") or "").strip(),
            (doc.get("image_ocr_text") or "").strip(),
            (doc.get("image_analysis_text") or "").strip(),
        ]
        text = "\n\n".join(part for part in text_parts if part)
        if not text:
            continue

        for idx, start in enumerate(range(0, len(text), stride)):
            piece = text[start : start + chunk_size].strip()
            if not piece:
                continue
            chunk = dict(doc)
            chunk["chunk_index"] = idx
            chunk["chunk_text"] = piece
            fingerprint_input = (
                f"{doc.get('source_type')}|{doc.get('issue_number')}|{doc.get('comment_id')}|"
                f"{doc.get('url')}|{doc.get('source_path')}|{doc.get('title')}"
            )
            source_fp = hashlib.sha1(fingerprint_input.encode("utf-8", errors="ignore")).hexdigest()[:10]
            chunk["chunk_id"] = (
                f"{doc.get('source_type')}:{doc.get('issue_number')}:{doc.get('comment_id')}:{source_fp}:{idx}"
            )
            chunks.append(chunk)

    return chunks

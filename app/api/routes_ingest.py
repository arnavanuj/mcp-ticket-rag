"""Ingestion API route."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import IngestRequest, IngestResponse
from app.services.ingest_service import run_ingest

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    settings = get_settings()
    settings.owner = payload.owner
    settings.repo = payload.repo
    settings.max_issues = payload.max_issues
    settings.include_comments = payload.include_comments
    settings.max_ocr_images = payload.max_images
    settings.include_repo_docs = payload.include_repo_docs

    result = run_ingest(settings)
    return IngestResponse(**result)

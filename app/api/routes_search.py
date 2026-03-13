"""Search API route."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.config import get_settings
from app.models.schemas import SearchResponse
from app.services.search_service import search_tickets

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1)) -> SearchResponse:
    settings = get_settings()
    results = search_tickets(settings=settings, query=q)
    return SearchResponse(query=q, results=results)

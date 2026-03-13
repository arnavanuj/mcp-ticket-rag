"""Ticket API route."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.models.schemas import TicketResponse
from app.services.ticket_service import get_ticket

router = APIRouter(tags=["ticket"])


@router.get("/ticket/{issue_number}", response_model=TicketResponse)
def ticket(issue_number: int) -> TicketResponse:
    settings = get_settings()
    try:
        data = get_ticket(settings=settings, issue_number=issue_number)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return TicketResponse(**data)

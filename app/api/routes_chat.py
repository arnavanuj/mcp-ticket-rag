"""Chat API route."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import chat

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat_route(payload: ChatRequest) -> ChatResponse:
    settings = get_settings()
    response = chat(settings=settings, query=payload.question, top_k=payload.top_k)
    return ChatResponse(**response)

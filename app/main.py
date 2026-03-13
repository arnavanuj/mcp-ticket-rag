"""FastAPI application bootstrap."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes_chat import router as chat_router
from app.api.routes_health import router as health_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_search import router as search_router
from app.api.routes_ticket import router as ticket_router

app = FastAPI(
    title="mcp-ticket-rag",
    version="0.2.0",
    description="MCP + lightweight RAG demo over GitHub issues and comments.",
)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(chat_router)
app.include_router(ticket_router)
app.include_router(search_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"project": "mcp-ticket-rag", "status": "ready"}

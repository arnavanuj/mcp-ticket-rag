"""Pydantic API and pipeline schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.citations import EvidenceItem


class HealthResponse(BaseModel):
    status: str
    project: str


class IngestRequest(BaseModel):
    owner: str = "langchain-ai"
    repo: str = "langchain"
    max_issues: int = Field(default=20, ge=1, le=50)
    include_comments: bool = True
    max_images: int = Field(default=10, ge=0, le=10)
    include_repo_docs: bool = False


class IngestResponse(BaseModel):
    owner: str
    repo: str
    mcp_connected: bool
    discovered_tools: list[str]
    used_tools: list[str]
    rest_fallback_used: bool
    zero_result_mcp_call: bool
    issues_ingested: int
    comments_ingested: int
    image_urls_detected: int
    supported_image_urls: int = 0
    skipped_image_urls: int = 0
    images_analyzed: int
    image_artifacts_created: int
    normalized_documents: int
    chunks_created: int
    chroma_vector_count: int


class ChatRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    evidence: list[EvidenceItem]
    issue_results: list[dict] = []
    route_used: str = "RAG"
    route_reason: str = ""
    mcp_tool_used: str = ""
    source_fields_used: list[str] = []
    source_field_paths_used: list[str] = []
    raw_evidence_snippets: list[str] = []
    confidence: float = 0.0
    llm_diagnostics: dict = {}


class TicketResponse(BaseModel):
    issue_number: int
    issue_title: str
    issue_url: str
    closed_at: str | None = None
    assignees: list[str]
    assignee_names: list[str] = []
    comments: list[dict]


class SearchResponse(BaseModel):
    query: str
    results: list[dict]

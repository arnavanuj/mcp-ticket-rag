"""Hybrid query router with semantic field resolution and explicit capability maps."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import Settings
from app.services.semantic_field_resolver import SemanticResolution, resolve_query_semantics

RAG_SCHEMA_FIELDS: set[str] = {
    "issue_number",
    "title",
    "raw_text",
    "labels",
    "assignees",
    "assignee_names",
    "comment_author",
    "comment_author_name",
    "created_at",
    "updated_at",
    "closed_at",
    "url",
    "image_ocr_text",
    "image_analysis_text",
    "author",
}

MCP_FIELD_CAPABILITIES: dict[str, list[str]] = {
    "assignees": ["issue_read:get"],
    "assignee": ["issue_read:get"],
    "assignee_names": ["issue_read:get"],
    "owner": ["issue_read:get"],
    "labels": ["issue_read:get"],
    "status": ["issue_read:get"],
    "state": ["issue_read:get"],
    "created_at": ["issue_read:get"],
    "opened_at": ["issue_read:get"],
    "opened": ["issue_read:get"],
    "start_time": ["issue_read:get"],
    "updated_at": ["issue_read:get"],
    "closed_at": ["issue_read:get"],
    "title": ["issue_read:get"],
    "body": ["issue_read:get"],
    "url": ["issue_read:get"],
    "author": ["issue_read:get"],
    "comment_author": ["issue_read:get_comments"],
    "comment_text": ["issue_read:get_comments"],
    "user.login": ["issue_read:get_comments"],
    "body_text": ["issue_read:get_comments"],
}


@dataclass(slots=True)
class RouteDecision:
    route_used: str
    route_reason: str
    intent_type: str
    issue_number: int | None
    detected_fields: list[str]
    rag_fields_available: bool
    mcp_fields_available: bool
    mcp_tools_for_fields: list[str]
    candidate_field_meanings: list[str]
    preferred_tool: str
    confidence: float
    reasoning: str
    semantic_resolver_diagnostics: dict


def route_query(settings: Settings, query: str, fresh_ingest_ran: bool = False) -> RouteDecision:
    """Route query to RAG, MCP Live, or RAG after fresh ingest."""

    issue_number = _extract_issue_number(query)
    resolution = resolve_query_semantics(settings=settings, query=query)

    candidate_fields = _normalize_candidates(resolution)
    rag_fields_available = any(field in RAG_SCHEMA_FIELDS for field in candidate_fields) if candidate_fields else True
    mcp_fields_available = any(field in MCP_FIELD_CAPABILITIES for field in candidate_fields) if candidate_fields else (
        resolution.preferred_tool != "rag"
    )
    mcp_tools_for_fields = _collect_tools(candidate_fields, resolution)

    if fresh_ingest_ran:
        return RouteDecision(
            route_used="RAG after Fresh Ingest",
            route_reason="Fresh ingest selected; route to indexed RAG after ingestion completes.",
            intent_type=resolution.intent_type,
            issue_number=issue_number,
            detected_fields=candidate_fields,
            rag_fields_available=rag_fields_available,
            mcp_fields_available=mcp_fields_available,
            mcp_tools_for_fields=mcp_tools_for_fields,
            candidate_field_meanings=candidate_fields,
            preferred_tool=resolution.preferred_tool,
            confidence=resolution.confidence,
            reasoning=resolution.reasoning,
            semantic_resolver_diagnostics=resolution.diagnostics,
        )

    if resolution.intent_type == "semantic":
        return RouteDecision(
            route_used="RAG",
            route_reason="Semantic/similarity/grouping/summarization query.",
            intent_type=resolution.intent_type,
            issue_number=issue_number,
            detected_fields=candidate_fields,
            rag_fields_available=rag_fields_available,
            mcp_fields_available=mcp_fields_available,
            mcp_tools_for_fields=mcp_tools_for_fields,
            candidate_field_meanings=candidate_fields,
            preferred_tool=resolution.preferred_tool,
            confidence=resolution.confidence,
            reasoning=resolution.reasoning,
            semantic_resolver_diagnostics=resolution.diagnostics,
        )

    if resolution.route_preference == "MCP Live" and mcp_fields_available and issue_number is not None:
        return RouteDecision(
            route_used="MCP Live",
            route_reason="Exact source-field lookup with MCP-capable field/tool and issue number.",
            intent_type=resolution.intent_type,
            issue_number=issue_number,
            detected_fields=candidate_fields,
            rag_fields_available=rag_fields_available,
            mcp_fields_available=mcp_fields_available,
            mcp_tools_for_fields=mcp_tools_for_fields,
            candidate_field_meanings=candidate_fields,
            preferred_tool=resolution.preferred_tool,
            confidence=resolution.confidence,
            reasoning=resolution.reasoning,
            semantic_resolver_diagnostics=resolution.diagnostics,
        )

    if rag_fields_available:
        return RouteDecision(
            route_used="RAG",
            route_reason="Exact query but target field family exists in current indexed RAG schema.",
            intent_type=resolution.intent_type,
            issue_number=issue_number,
            detected_fields=candidate_fields,
            rag_fields_available=rag_fields_available,
            mcp_fields_available=mcp_fields_available,
            mcp_tools_for_fields=mcp_tools_for_fields,
            candidate_field_meanings=candidate_fields,
            preferred_tool=resolution.preferred_tool,
            confidence=resolution.confidence,
            reasoning=resolution.reasoning,
            semantic_resolver_diagnostics=resolution.diagnostics,
        )

    return RouteDecision(
        route_used="MCP Live" if issue_number is not None else "RAG",
        route_reason="Fallback route after semantic resolution.",
        intent_type=resolution.intent_type,
        issue_number=issue_number,
        detected_fields=candidate_fields,
        rag_fields_available=rag_fields_available,
        mcp_fields_available=mcp_fields_available,
        mcp_tools_for_fields=mcp_tools_for_fields,
        candidate_field_meanings=candidate_fields,
        preferred_tool=resolution.preferred_tool,
        confidence=resolution.confidence,
        reasoning=resolution.reasoning,
        semantic_resolver_diagnostics=resolution.diagnostics,
    )


def _extract_issue_number(query: str) -> int | None:
    patterns = [
        r"(?:ticket|issue)\s*#?\s*(\d+)",
        r"#(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                continue
    return None


def _normalize_candidates(resolution: SemanticResolution) -> list[str]:
    candidates = [resolution.target_field_meaning] + list(resolution.equivalent_field_meanings)
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item).strip().lower().replace(" ", "_")
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _collect_tools(candidate_fields: list[str], resolution: SemanticResolution) -> list[str]:
    tools: list[str] = []
    for field in candidate_fields:
        for tool in MCP_FIELD_CAPABILITIES.get(field, []):
            if tool not in tools:
                tools.append(tool)
    if resolution.preferred_tool and resolution.preferred_tool != "rag" and resolution.preferred_tool not in tools:
        tools.append(resolution.preferred_tool)
    return tools

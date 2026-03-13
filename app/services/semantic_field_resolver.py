"""LLM-assisted semantic resolution for routing and live field lookup."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.ollama_client import OllamaClient, OllamaGenerationError

logger = get_logger("app.services.semantic_field_resolver")


@dataclass(slots=True)
class SemanticResolution:
    intent_type: str
    target_field_meaning: str
    equivalent_field_meanings: list[str]
    preferred_tool: str
    route_preference: str
    confidence: float
    reasoning: str
    diagnostics: dict[str, Any]


def resolve_query_semantics(settings: Settings, query: str) -> SemanticResolution:
    """Resolve user intent/field/tool semantically with Mistral; fallback safely."""

    seeded = _seed_resolution(query)
    if seeded is not None:
        return seeded

    fallback = _fallback_resolution(query)
    client = OllamaClient(base_url=settings.mistral_base_url, default_max_retries=settings.ollama_max_retries)

    prompt = (
        "You are a strict query-intent resolver for GitHub issue data.\n"
        "Return JSON only with keys:\n"
        "intent_type, target_field_meaning, equivalent_field_meanings, preferred_tool, route_preference, confidence, reasoning.\n"
        "Rules:\n"
        "- intent_type: exact_lookup or semantic\n"
        "- route_preference: MCP Live or RAG\n"
        "- preferred_tool: one of issue_read:get, issue_read:get_comments, rag\n"
        "- equivalent_field_meanings: list of canonical candidate fields\n"
        "- confidence: number 0..1\n"
        "- No markdown.\n\n"
        f"Query: {query}"
    )

    try:
        result = client.generate(
            component="semantic_resolver",
            model=settings.semantic_resolver_model,
            prompt=prompt,
            connect_timeout_seconds=settings.semantic_resolver_connect_timeout_seconds,
            first_byte_timeout_seconds=settings.semantic_resolver_first_byte_timeout_seconds,
            total_timeout_seconds=settings.semantic_resolver_timeout_seconds,
            max_retries=settings.ollama_max_retries,
            options={"temperature": 0.0},
        )
        raw = result.text
        parsed = _parse_json(raw)
        if not parsed:
            logger.warning("FIELD_RESOLUTION_FALLBACK | reason=llm_json_parse_failed | raw=%s", raw[:300])
            fallback.diagnostics = result.diagnostics
            return fallback
        return _coerce_resolution(parsed, fallback, diagnostics=result.diagnostics)
    except OllamaGenerationError as exc:
        logger.warning("FIELD_RESOLUTION_FALLBACK | reason=llm_call_failed | error=%s", exc)
        fallback.diagnostics = exc.diagnostics
        return fallback


def _parse_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to recover JSON object embedded in text.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _coerce_resolution(parsed: dict[str, Any], fallback: SemanticResolution, diagnostics: dict[str, Any]) -> SemanticResolution:
    intent_type = str(parsed.get("intent_type", fallback.intent_type)).strip().lower()
    if intent_type not in {"exact_lookup", "semantic"}:
        intent_type = fallback.intent_type

    route_preference = str(parsed.get("route_preference", fallback.route_preference)).strip()
    if route_preference not in {"MCP Live", "RAG"}:
        route_preference = fallback.route_preference

    preferred_tool = str(parsed.get("preferred_tool", fallback.preferred_tool)).strip()
    if preferred_tool not in {"issue_read:get", "issue_read:get_comments", "rag"}:
        preferred_tool = fallback.preferred_tool

    eq = parsed.get("equivalent_field_meanings", fallback.equivalent_field_meanings)
    equivalent = [str(x).strip() for x in eq] if isinstance(eq, list) else list(fallback.equivalent_field_meanings)
    equivalent = [x for x in equivalent if x]
    if not equivalent:
        equivalent = list(fallback.equivalent_field_meanings)

    confidence = parsed.get("confidence", fallback.confidence)
    try:
        confidence_num = float(confidence)
    except Exception:
        confidence_num = fallback.confidence
    confidence_num = max(0.0, min(1.0, confidence_num))

    return SemanticResolution(
        intent_type=intent_type,
        target_field_meaning=str(parsed.get("target_field_meaning", fallback.target_field_meaning)),
        equivalent_field_meanings=equivalent,
        preferred_tool=preferred_tool,
        route_preference=route_preference,
        confidence=confidence_num,
        reasoning=str(parsed.get("reasoning", fallback.reasoning)),
        diagnostics=diagnostics,
    )


def _fallback_resolution(query: str) -> SemanticResolution:
    q = query.lower()

    semantic_tokens = ("similar", "pattern", "summarize", "summary", "trend", "group", "compare", "discuss")
    if any(token in q for token in semantic_tokens):
        return SemanticResolution(
            intent_type="semantic",
            target_field_meaning="semantic_reasoning",
            equivalent_field_meanings=["raw_text", "title", "image_analysis_text"],
            preferred_tool="rag",
            route_preference="RAG",
            confidence=0.55,
            reasoning="Fallback heuristic classified query as semantic reasoning/search.",
            diagnostics={},
        )

    if "comment" in q or "commented" in q:
        return SemanticResolution(
            intent_type="exact_lookup",
            target_field_meaning="comment_author",
            equivalent_field_meanings=["comment_author", "author_login", "user.login", "comment_text", "body"],
            preferred_tool="issue_read:get_comments",
            route_preference="MCP Live",
            confidence=0.6,
            reasoning="Fallback heuristic detected commenter lookup.",
            diagnostics={},
        )

    if ("who opened" in q) or ("who started" in q and "when" not in q):
        return SemanticResolution(
            intent_type="exact_lookup",
            target_field_meaning="author",
            equivalent_field_meanings=["author", "user.login", "user.name", "opened_by", "creator"],
            preferred_tool="issue_read:get",
            route_preference="MCP Live",
            confidence=0.6,
            reasoning="Fallback heuristic mapped opener question to author/creator fields.",
            diagnostics={},
        )

    if any(token in q for token in ("assignee", "assigned", "sme")):
        return SemanticResolution(
            intent_type="exact_lookup",
            target_field_meaning="assignees",
            equivalent_field_meanings=["assignees", "assignee", "owner", "assignees.login", "assignees.name"],
            preferred_tool="issue_read:get",
            route_preference="MCP Live",
            confidence=0.6,
            reasoning="Fallback heuristic detected assignee lookup.",
            diagnostics={},
        )

    if any(token in q for token in ("opened", "started", "initiated", "created", "raised")):
        return SemanticResolution(
            intent_type="exact_lookup",
            target_field_meaning="created_at",
            equivalent_field_meanings=["created_at", "opened_at", "created", "start_time", "initiated_at"],
            preferred_tool="issue_read:get",
            route_preference="MCP Live",
            confidence=0.6,
            reasoning="Fallback heuristic mapped open/start/initiate language to creation timestamp.",
            diagnostics={},
        )

    if "label" in q:
        return SemanticResolution(
            intent_type="exact_lookup",
            target_field_meaning="labels",
            equivalent_field_meanings=["labels", "label", "tags"],
            preferred_tool="issue_read:get",
            route_preference="MCP Live",
            confidence=0.6,
            reasoning="Fallback heuristic detected labels lookup.",
            diagnostics={},
        )

    return SemanticResolution(
        intent_type="semantic",
        target_field_meaning="semantic_reasoning",
        equivalent_field_meanings=["raw_text", "title"],
        preferred_tool="rag",
        route_preference="RAG",
        confidence=0.5,
        reasoning="Fallback default to RAG semantic path.",
        diagnostics={},
    )


def _seed_resolution(query: str) -> SemanticResolution | None:
    """Small deterministic seed rules for explicit exact-lookups."""

    q = query.lower().strip()

    if "who commented that" in q and ("ticket" in q or "issue" in q):
        return SemanticResolution(
            intent_type="exact_lookup",
            target_field_meaning="comment_author",
            equivalent_field_meanings=["comment_author", "user.login", "body", "comment_text"],
            preferred_tool="issue_read:get_comments",
            route_preference="MCP Live",
            confidence=0.8,
            reasoning="Seed rule: explicit commenter lookup with quoted/targeted comment phrase.",
            diagnostics={},
        )

    if any(token in q for token in ("who is assignee", "assignee of issue", "sme for issue", "sme on ticket")):
        return SemanticResolution(
            intent_type="exact_lookup",
            target_field_meaning="assignees",
            equivalent_field_meanings=["assignees", "assignee", "owner", "assignees.login", "assignees.name"],
            preferred_tool="issue_read:get",
            route_preference="MCP Live",
            confidence=0.78,
            reasoning="Seed rule: explicit assignee/SME lookup.",
            diagnostics={},
        )

    if "who opened issue" in q or "who started ticket" in q:
        return SemanticResolution(
            intent_type="exact_lookup",
            target_field_meaning="author",
            equivalent_field_meanings=["author", "user.login", "user.name", "opened_by", "creator"],
            preferred_tool="issue_read:get",
            route_preference="MCP Live",
            confidence=0.78,
            reasoning="Seed rule: explicit opener/creator lookup.",
            diagnostics={},
        )

    if ("when was issue" in q or "when was ticket" in q) and any(
        token in q for token in ("opened", "started", "initiated", "raised")
    ):
        return SemanticResolution(
            intent_type="exact_lookup",
            target_field_meaning="created_at",
            equivalent_field_meanings=["created_at", "opened_at", "created", "start_time", "initiated_at"],
            preferred_tool="issue_read:get",
            route_preference="MCP Live",
            confidence=0.78,
            reasoning="Seed rule: open/start/initiate timestamp lookup.",
            diagnostics={},
        )

    if ("what labels" in q or "labels are on issue" in q or "labels are on ticket" in q):
        return SemanticResolution(
            intent_type="exact_lookup",
            target_field_meaning="labels",
            equivalent_field_meanings=["labels", "label", "tags"],
            preferred_tool="issue_read:get",
            route_preference="MCP Live",
            confidence=0.78,
            reasoning="Seed rule: explicit labels lookup.",
            diagnostics={},
        )

    return None

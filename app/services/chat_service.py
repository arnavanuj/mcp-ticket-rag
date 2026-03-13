"""Chat service with hybrid routing between RAG and MCP Live lookup."""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger, timed_step
from app.mcp.github_client import GitHubMCPClient
from app.rag.answer_builder import AnswerBuilder
from app.rag.embedder import Embedder
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.services.ollama_client import OllamaClient, OllamaGenerationError
from app.services.payload_introspector import FieldEntry, extract_best_value, flatten_payload, search_payload_fields
from app.services.query_router import RouteDecision, route_query

logger = get_logger("app.services.chat_service")


def chat(
    settings: Settings,
    query: str,
    top_k: int | None = None,
    fresh_ingest_ran: bool = False,
) -> dict[str, Any]:
    """Route query to RAG or MCP Live, then return grounded answer + evidence."""

    decision = route_query(settings=settings, query=query, fresh_ingest_ran=fresh_ingest_ran)
    logger.info(
        "FIELD_RESOLUTION | intent=%s | target_fields=%s | preferred_tool=%s | confidence=%.2f | reasoning=%s",
        decision.intent_type,
        decision.candidate_field_meanings,
        decision.preferred_tool,
        decision.confidence,
        decision.reasoning,
    )
    logger.info(
        "ROUTE_DECISION | route=%s | reason=%s | intent=%s | issue_number=%s | fields=%s | rag_fields_available=%s | mcp_fields_available=%s | mcp_tools=%s",
        decision.route_used,
        decision.route_reason,
        decision.intent_type,
        decision.issue_number,
        decision.detected_fields,
        decision.rag_fields_available,
        decision.mcp_fields_available,
        decision.mcp_tools_for_fields,
    )

    if decision.route_used == "MCP Live":
        return _chat_mcp_live(settings=settings, query=query, decision=decision)

    return _chat_rag(
        settings=settings,
        query=query,
        top_k=top_k,
        route_used=decision.route_used,
        route_reason=decision.route_reason,
        confidence=decision.confidence,
        semantic_resolver_diagnostics=decision.semantic_resolver_diagnostics,
    )


def _chat_rag(
    settings: Settings,
    query: str,
    top_k: int | None,
    route_used: str,
    route_reason: str,
    confidence: float,
    semantic_resolver_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    k = top_k or settings.retrieval_top_k
    if "similar issues" in query.lower():
        k = max(k, 15)

    embedder = Embedder(settings=settings)
    vector_store = VectorStore(settings=settings)
    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    answer_builder = AnswerBuilder(settings=settings)

    with timed_step(logger, "retrieve", "Retrieve top semantic matches from vector store"):
        retrieved = retriever.retrieve(query=query, top_k=k)
        matched_issues = sorted({item.get("issue_number") for item in retrieved if item.get("issue_number") is not None})
        matched_urls = [item.get("url") for item in retrieved if item.get("url")]
        top_chunks = [
            {
                "issue_number": item.get("issue_number"),
                "source_type": item.get("source_type"),
                "assignees": item.get("assignees"),
                "comment_author": item.get("comment_author"),
                "snippet": (item.get("chunk_text") or "")[:160],
            }
            for item in retrieved
        ]
        logger.info(
            "RETRIEVE_RESULT | question=%s | top_k=%s | matched_issues=%s | matched_urls=%s | top_chunks=%s",
            query,
            k,
            matched_issues,
            matched_urls,
            top_chunks,
        )

    with timed_step(logger, "answer_generation", "Call local Mistral with retrieved evidence context"):
        response = answer_builder.build_answer(question=query, retrieved=retrieved)

    response["route_used"] = route_used
    response["route_reason"] = route_reason
    response["mcp_tool_used"] = ""
    response["source_fields_used"] = []
    response["source_field_paths_used"] = []
    response["raw_evidence_snippets"] = []
    response["confidence"] = confidence
    response["llm_diagnostics"] = {
        "semantic_resolver": semantic_resolver_diagnostics,
        "answer_generation": response.get("llm_diagnostics", {}).get("answer_generation", []),
    }
    return response


def _chat_mcp_live(settings: Settings, query: str, decision: RouteDecision) -> dict[str, Any]:
    """Direct MCP lookup path for exact field-based grounded answers."""

    logger.info("VECTOR_DB_UNCHANGED | mode=mcp_live | note=no_vector_upsert_performed")
    issue_number = decision.issue_number
    if issue_number is None:
        return {
            "answer": "not clearly stated in the source payload",
            "evidence": [],
            "issue_results": [],
            "route_used": "MCP Live",
            "route_reason": "Exact lookup detected but issue number missing.",
            "mcp_tool_used": "",
            "source_fields_used": decision.candidate_field_meanings,
            "source_field_paths_used": [],
            "raw_evidence_snippets": [],
            "confidence": decision.confidence,
            "llm_diagnostics": {"semantic_resolver": decision.semantic_resolver_diagnostics, "answer_generation": []},
        }

    client = GitHubMCPClient(settings=settings)
    query_lower = query.lower()
    is_comment_lookup = "comment" in query_lower or "commented" in query_lower or decision.preferred_tool.endswith("get_comments")

    if is_comment_lookup:
        raw_comments = client.get_issue_comments_raw(owner=settings.owner, repo=settings.repo, issue_number=issue_number)
        return _answer_from_comments(settings=settings, query=query, decision=decision, raw_comments=raw_comments, issue_number=issue_number)

    raw_issue = client.get_issue_raw(owner=settings.owner, repo=settings.repo, issue_number=issue_number)
    entries = flatten_payload(raw_issue)

    if _is_assignee_query(query_lower):
        return _answer_assignee_query(raw_issue=raw_issue, entries=entries, issue_number=issue_number, decision=decision)

    best = extract_best_value(raw_issue, candidate_field_meanings=decision.candidate_field_meanings)
    if not best:
        return _mcp_empty(decision=decision, issue_number=issue_number, tool_used="issue_read(get)")

    logger.info(
        "MCP_DYNAMIC_FIELD_MATCH | issue=%s | tool=%s | matched_path=%s | score=%.3f",
        issue_number,
        "issue_read(get)",
        best["path"],
        float(best.get("score", 0.0)),
    )

    label = _humanize_target(decision.candidate_field_meanings)
    answer = f"Issue #{issue_number} {label}: {best['value']}"
    snippet = f"{best['path']} = {best['value']}"

    return _mcp_response(
        answer=answer,
        issue_number=issue_number,
        issue_title=str(raw_issue.get("title", "")),
        issue_url=str(raw_issue.get("html_url") or raw_issue.get("url") or ""),
        route_reason=decision.route_reason,
        mcp_tool_used="issue_read(get)",
        source_fields_used=decision.candidate_field_meanings,
        source_field_paths_used=[best["path"]],
        raw_evidence_snippets=[snippet],
        snippet=snippet,
        confidence=decision.confidence,
        semantic_resolver_diagnostics=decision.semantic_resolver_diagnostics,
        extra_llm_diagnostics=[],
    )


def _answer_assignee_query(
    raw_issue: dict[str, Any],
    entries: list[FieldEntry],
    issue_number: int,
    decision: RouteDecision,
) -> dict[str, Any]:
    assignee_entries = [
        e for e in entries if "assignee" in e.path.lower() and e.key.lower() in {"login", "name", "assignee", "assignees"}
    ]
    if not assignee_entries:
        assignee_entries = [e for e in entries if "assignee" in e.path.lower()]

    values = [e.normalized_text_value for e in assignee_entries if e.normalized_text_value]
    values = [v for v in values if v not in {"", "none", "null"}]
    unique_values = sorted(set(values))

    if not unique_values:
        answer = f"Issue #{issue_number} assignee: not clearly stated in the source payload"
    else:
        answer = f"Issue #{issue_number} assignee(s): {', '.join(unique_values)}"

    paths = [e.path for e in assignee_entries[:8]]
    snippets = [f"{e.path} = {e.value}" for e in assignee_entries[:8]]
    logger.info("MCP_DYNAMIC_FIELD_MATCH | issue=%s | tool=%s | matched_paths=%s", issue_number, "issue_read(get)", paths)

    return _mcp_response(
        answer=answer,
        issue_number=issue_number,
        issue_title=str(raw_issue.get("title", "")),
        issue_url=str(raw_issue.get("html_url") or raw_issue.get("url") or ""),
        route_reason=decision.route_reason,
        mcp_tool_used="issue_read(get)",
        source_fields_used=decision.candidate_field_meanings,
        source_field_paths_used=paths,
        raw_evidence_snippets=snippets,
        snippet="; ".join(snippets[:2]) if snippets else "not clearly stated in the source payload",
        confidence=decision.confidence,
        semantic_resolver_diagnostics=decision.semantic_resolver_diagnostics,
        extra_llm_diagnostics=[],
    )


def _answer_from_comments(
    settings: Settings,
    query: str,
    decision: RouteDecision,
    raw_comments: list[dict[str, Any]],
    issue_number: int,
) -> dict[str, Any]:
    if not raw_comments:
        return _mcp_empty(decision=decision, issue_number=issue_number, tool_used="issue_read(get_comments)")

    phrase = _extract_comment_phrase(query)
    coarse = _coarse_filter_comments(raw_comments, phrase)
    ranked_index, rank_confidence, _, rank_diag = _rank_comments_with_model(settings=settings, query=query, comments=coarse)
    selected = coarse[ranked_index] if coarse else raw_comments[0]

    flat = flatten_payload(selected)
    author_candidate = extract_best_value(selected, ["user.login", "author", "comment_author", "user.name"])
    body_candidate = extract_best_value(selected, ["body", "comment_text", "raw_text"])

    author = (author_candidate or {}).get("value") or "not clearly stated in the source payload"
    snippet = str((body_candidate or {}).get("value") or "")[:280]

    paths = []
    if author_candidate:
        paths.append(author_candidate["path"])
    if body_candidate:
        paths.append(body_candidate["path"])

    raw_snippets = [f"{entry.path} = {entry.value}" for entry in flat if entry.path in paths]
    logger.info(
        "MCP_DYNAMIC_FIELD_MATCH | issue=%s | tool=%s | matched_paths=%s | ranked_confidence=%.2f",
        issue_number,
        "issue_read(get_comments)",
        paths,
        rank_confidence,
    )

    url = str(selected.get("html_url") or selected.get("url") or "")
    return {
        "answer": f"Matched commenter on issue #{issue_number}: {author}",
        "evidence": [
            {
                "source_type": "comment",
                "issue_number": issue_number,
                "comment_id": selected.get("id"),
                "title": f"Issue #{issue_number}",
                "url": url,
                "comment_author": str(author),
                "snippet": snippet or "not clearly stated in the source payload",
            }
        ],
        "issue_results": [],
        "route_used": "MCP Live",
        "route_reason": decision.route_reason,
        "mcp_tool_used": "issue_read(get_comments)",
        "source_fields_used": decision.candidate_field_meanings,
        "source_field_paths_used": paths,
        "raw_evidence_snippets": raw_snippets,
        "confidence": max(decision.confidence, rank_confidence),
        "llm_diagnostics": {
            "semantic_resolver": decision.semantic_resolver_diagnostics,
            "answer_generation": [rank_diag] if rank_diag else [],
        },
    }


def _mcp_response(
    answer: str,
    issue_number: int,
    issue_title: str,
    issue_url: str,
    route_reason: str,
    mcp_tool_used: str,
    source_fields_used: list[str],
    source_field_paths_used: list[str],
    raw_evidence_snippets: list[str],
    snippet: str,
    confidence: float,
    semantic_resolver_diagnostics: dict[str, Any],
    extra_llm_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "answer": answer,
        "evidence": [
            {
                "source_type": "issue",
                "issue_number": issue_number,
                "comment_id": None,
                "title": issue_title,
                "url": issue_url,
                "snippet": snippet or "not clearly stated in the source payload",
            }
        ],
        "issue_results": [],
        "route_used": "MCP Live",
        "route_reason": route_reason,
        "mcp_tool_used": mcp_tool_used,
        "source_fields_used": source_fields_used,
        "source_field_paths_used": source_field_paths_used,
        "raw_evidence_snippets": raw_evidence_snippets,
        "confidence": confidence,
        "llm_diagnostics": {
            "semantic_resolver": semantic_resolver_diagnostics,
            "answer_generation": extra_llm_diagnostics,
        },
    }


def _mcp_empty(decision: RouteDecision, issue_number: int, tool_used: str) -> dict[str, Any]:
    return {
        "answer": "not clearly stated in the source payload",
        "evidence": [
            {
                "source_type": "issue",
                "issue_number": issue_number,
                "comment_id": None,
                "title": f"Issue #{issue_number}",
                "url": "",
                "snippet": "not clearly stated in the source payload",
            }
        ],
        "issue_results": [],
        "route_used": "MCP Live",
        "route_reason": decision.route_reason,
        "mcp_tool_used": tool_used,
        "source_fields_used": decision.candidate_field_meanings,
        "source_field_paths_used": [],
        "raw_evidence_snippets": [],
        "confidence": decision.confidence,
        "llm_diagnostics": {"semantic_resolver": decision.semantic_resolver_diagnostics, "answer_generation": []},
    }


def _is_assignee_query(query_lower: str) -> bool:
    return any(token in query_lower for token in ("assignee", "assigned", "sme"))


def _humanize_target(fields: list[str]) -> str:
    for field in fields:
        if field in {"created_at", "opened_at", "opened", "start_time", "initiated_at"}:
            return "opened/created time"
        if field in {"author", "opened_by", "user.login", "creator"}:
            return "opened by"
        if field in {"labels", "label"}:
            return "labels"
        if field in {"closed_at"}:
            return "closed time"
    return fields[0] if fields else "value"


def _extract_comment_phrase(query: str) -> str:
    match = re.search(
        r"who commented that (.+?) (?:for|on) (?:ticket|issue)\s*#?\d+",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group(1).strip().strip("\"'").lower()


def _coarse_filter_comments(comments: list[dict[str, Any]], phrase: str) -> list[dict[str, Any]]:
    if not phrase:
        return comments[:10]
    tokens = [tok for tok in re.findall(r"[a-z0-9]+", phrase.lower()) if len(tok) > 2]
    if not tokens:
        return comments[:10]

    scored: list[tuple[int, dict[str, Any]]] = []
    for comment in comments:
        body = str(comment.get("body") or "").lower()
        overlap = sum(1 for tok in tokens if tok in body)
        if overlap > 0:
            scored.append((overlap, comment))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return comments[:10]
    return [comment for _, comment in scored[:10]]


def _rank_comments_with_model(
    settings: Settings,
    query: str,
    comments: list[dict[str, Any]],
) -> tuple[int, float, str, dict[str, Any] | None]:
    if not comments:
        return 0, 0.0, "No comments", None

    compact = []
    for idx, comment in enumerate(comments):
        compact.append(
            {
                "idx": idx,
                "author": (comment.get("user") or {}).get("login") if isinstance(comment.get("user"), dict) else "",
                "body": str(comment.get("body") or "")[:400],
            }
        )

    prompt = (
        "Select the single best matching comment for the query. Return JSON only with keys: best_index, confidence, reasoning.\n"
        f"Query: {query}\n"
        f"Comments: {json.dumps(compact, ensure_ascii=False)}"
    )
    client = OllamaClient(base_url=settings.mistral_base_url, default_max_retries=settings.ollama_max_retries)
    try:
        result = client.generate(
            component="semantic_resolver_comment_rank",
            model=settings.semantic_resolver_model,
            prompt=prompt,
            connect_timeout_seconds=settings.semantic_resolver_connect_timeout_seconds,
            first_byte_timeout_seconds=settings.semantic_resolver_first_byte_timeout_seconds,
            total_timeout_seconds=settings.semantic_resolver_timeout_seconds,
            max_retries=settings.ollama_max_retries,
            options={"temperature": 0.0},
        )
        text = result.text
        parsed = _parse_json(text)
        if not isinstance(parsed, dict):
            return 0, 0.5, "LLM parse failed", result.diagnostics
        idx = int(parsed.get("best_index", 0))
        idx = max(0, min(idx, len(comments) - 1))
        conf = float(parsed.get("confidence", 0.5))
        conf = max(0.0, min(conf, 1.0))
        reason = str(parsed.get("reasoning", ""))
        return idx, conf, reason, result.diagnostics
    except OllamaGenerationError as exc:
        logger.warning("MCP_COMMENT_RANK_FALLBACK | reason=%s", exc.diagnostics.get("fallback_reason", str(exc)))
        return 0, 0.45, "LLM ranking failed", exc.diagnostics


def _parse_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

"""Streamlit UI for manual MCP-first ingestion + RAG querying."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.chat_service import chat
from app.services.ingest_service import run_ingest
from app.services.ollama_client import OllamaClient


def _fmt_seconds(value: float) -> str:
    return f"{value:.2f}s"


def _render_issue_results(issue_results: list[dict[str, Any]]) -> None:
    if not issue_results:
        st.info("No structured issue results were returned for this query.")
        return

    st.subheader("Issue Results")
    for idx, item in enumerate(issue_results, start=1):
        title = item.get("title") or "(no title)"
        with st.expander(f"{idx}. #{item.get('issue_id')} - {title}", expanded=(idx == 1)):
            st.markdown(f"**Issue ID:** {item.get('issue_id')}")
            st.markdown(f"**Issue URL:** {item.get('issue_url') or 'not available'}")
            st.markdown(f"**Open Duration:** {item.get('open_duration') or 'not clearly stated'}")
            st.markdown(f"**Resolved By:** {item.get('resolved_by') or 'not clearly stated in the issue/comments'}")
            st.markdown(f"**Summary:** {item.get('summary') or 'not clearly stated'}")
            st.markdown(f"**Resolution:** {item.get('resolution') or 'not clearly stated'}")
            snippets = item.get("evidence") or []
            if snippets:
                st.markdown("**Evidence Snippets**")
                for snippet in snippets:
                    st.code(str(snippet), language="text")


def _render_evidence(evidence: list[dict[str, Any]]) -> None:
    st.subheader("Evidence")
    if not evidence:
        st.info("No evidence returned.")
        return

    for idx, ev in enumerate(evidence, start=1):
        with st.expander(f"{idx}. {ev.get('source_type', 'unknown')} | issue #{ev.get('issue_number')}", expanded=False):
            st.markdown(f"**Title:** {ev.get('title') or 'not available'}")
            st.markdown(f"**URL:** {ev.get('url') or 'not available'}")
            st.markdown(f"**Comment ID:** {ev.get('comment_id') or 'n/a'}")
            st.code(str(ev.get("snippet") or ""), language="text")


def main() -> None:
    st.set_page_config(page_title="mcp-ticket-rag", layout="wide")
    st.title("mcp-ticket-rag")
    st.caption("MCP-first GitHub issue RAG demo (LangChain issues + comments + selective image analysis)")

    settings = get_settings()
    if settings.ollama_enable_warmup and settings.ollama_warmup_on_start and not st.session_state.get("ollama_warmed"):
        with st.spinner("Running optional Ollama warmup..."):
            client = OllamaClient(base_url=settings.mistral_base_url, default_max_retries=settings.ollama_max_retries)
            warmup_logs = {
                "semantic_resolver": client.warmup_model(settings.semantic_resolver_model),
                "answer_model": client.warmup_model(settings.answer_model),
            }
            st.session_state["ollama_warmed"] = True
            st.session_state["ollama_warmup_logs"] = warmup_logs

    with st.sidebar:
        st.header("Runtime")
        st.write(f"Owner/Repo: `{settings.owner}/{settings.repo}`")
        st.write(f"Max Issues: `{settings.max_issues}`")
        st.write(f"Max OCR Images: `{settings.max_ocr_images}`")
        st.write(f"Vision Model: `{settings.phi3_vision_model}`")
        st.write(f"Semantic Resolver Model: `{settings.semantic_resolver_model}`")
        st.write(f"Answer Model: `{settings.answer_model}`")
        st.write(f"MCP Transport: `{settings.github_mcp_transport}`")
        if st.session_state.get("ollama_warmup_logs"):
            st.caption("Ollama warmup diagnostics available in session state.")

    query = st.text_area(
        "Query",
        value="Summarize the issue about adding token usage tracking in middleware",
        height=120,
        help="Enter any question over the currently indexed GitHub issues/comments corpus.",
    )
    run_fresh_ingest = st.checkbox("Run fresh ingest before query", value=False)

    if st.button("Run Query", type="primary", use_container_width=True):
        if not query.strip():
            st.error("Please enter a query.")
            return

        ingest_result: dict[str, Any] | None = None
        chat_result: dict[str, Any] | None = None

        ingest_seconds = 0.0
        query_seconds = 0.0
        total_start = time.perf_counter()

        try:
            if run_fresh_ingest:
                st.info("Starting ingestion...")
                ingest_start = time.perf_counter()
                with st.spinner("Running MCP-first ingestion and indexing..."):
                    ingest_result = run_ingest(settings)
                ingest_seconds = time.perf_counter() - ingest_start
                st.success("Ingestion completed.")

            st.info("Running query...")
            query_start = time.perf_counter()
            with st.spinner("Retrieving evidence and generating answer..."):
                chat_result = chat(settings=settings, query=query, fresh_ingest_ran=run_fresh_ingest)
            query_seconds = time.perf_counter() - query_start
            st.success("Completed.")

        except Exception as exc:
            st.error(f"Execution failed: {exc}")
            with st.expander("Error details", expanded=False):
                st.code(traceback.format_exc(), language="text")
            return

        total_seconds = time.perf_counter() - total_start

        st.subheader("Answer")
        st.write((chat_result or {}).get("answer", "No answer returned."))
        st.markdown(f"**Route Used:** {(chat_result or {}).get('route_used', 'unknown')}")
        st.markdown(f"**Route Reason:** {(chat_result or {}).get('route_reason', '')}")
        if (chat_result or {}).get("mcp_tool_used"):
            st.markdown(f"**MCP Tool Used:** {(chat_result or {}).get('mcp_tool_used')}")
        source_fields = (chat_result or {}).get("source_fields_used", [])
        if source_fields:
            st.markdown(f"**Source Fields Used:** {', '.join(source_fields)}")
        source_paths = (chat_result or {}).get("source_field_paths_used", [])
        if source_paths:
            st.markdown(f"**Source Field Paths Used:** {', '.join(source_paths)}")
        confidence = (chat_result or {}).get("confidence")
        if confidence is not None:
            st.markdown(f"**Confidence:** {confidence}")
        raw_snippets = (chat_result or {}).get("raw_evidence_snippets", [])
        if raw_snippets:
            st.markdown("**Raw Evidence Snippets**")
            for item in raw_snippets:
                st.code(str(item), language="text")

        st.subheader("LLM Execution Diagnostics")
        llm_diag = (chat_result or {}).get("llm_diagnostics", {})
        if not llm_diag:
            st.info("No LLM diagnostics available for this response.")
        else:
            st.json(llm_diag)

        _render_issue_results((chat_result or {}).get("issue_results", []))
        _render_evidence((chat_result or {}).get("evidence", []))

        st.subheader("Execution Timing")
        col1, col2, col3 = st.columns(3)
        col1.metric("Ingestion Time", _fmt_seconds(ingest_seconds))
        col2.metric("Query Time", _fmt_seconds(query_seconds))
        col3.metric("Total Time", _fmt_seconds(total_seconds))

        if ingest_result is not None:
            st.subheader("Ingestion Summary")
            st.json(ingest_result)


if __name__ == "__main__":
    main()

"""Run ingestion pipeline then execute a fixed RAG question."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.core.logging import get_logger, timed_step
from app.services.chat_service import chat
from app.services.ingest_service import run_ingest

TEST_QUERY = "Summarize the issue about adding token usage tracking in middleware"


def main() -> None:
    logger = get_logger("scripts.bootstrap_ingest")
    settings = get_settings()

    logger.info(
        "PIPELINE_CONFIG | owner=%s | repo=%s | max_issues=%s | max_ocr_images=%s | top_k=%s | mistral_endpoint=%s",
        settings.owner,
        settings.repo,
        settings.max_issues,
        settings.max_ocr_images,
        settings.retrieval_top_k,
        settings.mistral_base_url,
    )

    with timed_step(logger, "pipeline_ingest", "Execute issue/comment ingestion and vector indexing"):
        ingest_result = run_ingest(settings)

    with timed_step(logger, "pipeline_query", "Run fixed RAG test question"):
        chat_result = chat(settings=settings, query=TEST_QUERY, top_k=settings.retrieval_top_k)

    final = {
        "files_modified": [
            "app/core/config.py",
            "app/core/logging.py",
            "app/mcp/github_client.py",
            "app/mcp/adapters.py",
            "app/ingestion/issue_ingestor.py",
            "app/ingestion/comment_ingestor.py",
            "app/ingestion/asset_extractor.py",
            "app/ingestion/image_ocr.py",
            "app/ingestion/normalizer.py",
            "app/rag/chunker.py",
            "app/rag/embedder.py",
            "app/rag/vector_store.py",
            "app/rag/retriever.py",
            "app/rag/answer_builder.py",
            "app/services/ingest_service.py",
            "app/services/chat_service.py",
            "app/services/ticket_service.py",
            "app/services/search_service.py",
            "app/models/schemas.py",
            "app/models/citations.py",
            "app/api/routes_health.py",
            "app/api/routes_ingest.py",
            "app/api/routes_chat.py",
            "app/api/routes_ticket.py",
            "app/api/routes_search.py",
            "app/main.py",
            "scripts/bootstrap_ingest.py",
            "requirements.txt",
            ".env.example",
            "README.md",
        ],
        **ingest_result,
        "query": TEST_QUERY,
        "final_answer": chat_result.get("answer"),
        "evidence": chat_result.get("evidence", []),
    }

    logger.info("FINAL_OUTPUT\n%s", json.dumps(final, indent=2, ensure_ascii=False))
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

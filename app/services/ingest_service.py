"""Pipeline service: ingest issues/comments, multimodal analysis, chunk, embed, and index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger, timed_step
from app.ingestion.asset_extractor import SUPPORTED_IMAGE_EXTENSIONS, extract_image_urls, is_supported_image_url
from app.ingestion.comment_ingestor import ingest_comments
from app.ingestion.image_ocr import analyze_image_with_phi3, download_image
from app.ingestion.issue_ingestor import ingest_issues
from app.ingestion.normalizer import (
    normalize_comment,
    normalize_image_analysis,
    normalize_image_ocr,
    normalize_issue,
)
from app.mcp.github_client import GitHubMCPClient
from app.rag.chunker import chunk_documents
from app.rag.embedder import Embedder
from app.rag.vector_store import VectorStore

logger = get_logger("app.services.ingest_service")


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_ingest(settings: Settings) -> dict[str, Any]:
    """Execute full ingestion and indexing pipeline."""

    client = GitHubMCPClient(settings=settings)
    embedder = Embedder(settings=settings)
    vector_store = VectorStore(settings=settings)

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    with timed_step(logger, "fetch_issues", "Fetch latest closed issues via MCP-first access"):
        issues = ingest_issues(
            client=client,
            owner=settings.owner,
            repo=settings.repo,
            max_issues=settings.max_issues,
        )

    all_comments: dict[int, list[dict[str, Any]]] = {}
    total_comments = 0
    total_image_urls = 0
    accepted_image_urls = 0
    skipped_image_urls = 0
    image_records: list[tuple[int, str, str]] = []

    with timed_step(logger, "fetch_comments_and_assets", "Fetch comments and detect image URLs"):
        for issue in issues:
            issue_number = issue["issue_number"]
            issue_images = extract_image_urls(issue.get("issue_body", ""))
            total_image_urls += len(issue_images)
            for url in issue_images:
                if is_supported_image_url(url):
                    accepted_image_urls += 1
                    image_records.append((issue_number, issue.get("issue_title", ""), url))
                    logger.info("IMAGE_URL_ACCEPTED | issue=%s | url=%s | supported_ext=%s", issue_number, url, SUPPORTED_IMAGE_EXTENSIONS)
                else:
                    skipped_image_urls += 1
                    logger.info("IMAGE_URL_SKIPPED | issue=%s | url=%s | reason=unsupported_extension", issue_number, url)

            comments = []
            if settings.include_comments:
                comments = ingest_comments(
                    client=client,
                    owner=settings.owner,
                    repo=settings.repo,
                    issue_number=issue_number,
                )
                for comment in comments:
                    comment_images = extract_image_urls(comment.get("raw_text", ""))
                    total_image_urls += len(comment_images)
                    for url in comment_images:
                        if is_supported_image_url(url):
                            accepted_image_urls += 1
                            image_records.append((issue_number, issue.get("issue_title", ""), url))
                            logger.info(
                                "IMAGE_URL_ACCEPTED | issue=%s | comment_id=%s | url=%s | supported_ext=%s",
                                issue_number,
                                comment.get("comment_id"),
                                url,
                                SUPPORTED_IMAGE_EXTENSIONS,
                            )
                        else:
                            skipped_image_urls += 1
                            logger.info(
                                "IMAGE_URL_SKIPPED | issue=%s | comment_id=%s | url=%s | reason=unsupported_extension",
                                issue_number,
                                comment.get("comment_id"),
                                url,
                            )

            all_comments[issue_number] = comments
            total_comments += len(comments)

            assignees = issue.get("assignees") or ["unassigned"]
            assignee_names = issue.get("assignee_names") or []
            commenters = sorted({(c.get("author_login") or c.get("author") or "unknown") for c in comments})
            commenter_names = sorted({(c.get("author_name") or "") for c in comments if c.get("author_name")})
            logger.info(
                "ASSIGNEE_METADATA_CAPTURED | issue=%s | assignees=%s | assignee_names=%s",
                issue_number,
                assignees,
                assignee_names,
            )
            logger.info(
                "COMMENT_AUTHOR_METADATA_CAPTURED | issue=%s | commenter_usernames=%s | commenter_names=%s",
                issue_number,
                commenters,
                commenter_names,
            )
            image_count = len(issue_images) + sum(len(extract_image_urls(c.get("raw_text", ""))) for c in comments)
            logger.info(
                "ISSUE_FETCHED | number=%s | title=%s | url=%s | closed_at=%s | assignees=%s | comments=%s | image_urls=%s",
                issue_number,
                issue.get("issue_title"),
                issue.get("url"),
                issue.get("closed_at"),
                assignees,
                len(comments),
                image_count,
            )

    normalized_docs: list[dict[str, Any]] = []

    with timed_step(logger, "normalize_text_docs", "Normalize issues and comments into internal schema"):
        for issue in issues:
            normalized_docs.append(normalize_issue(settings.owner, settings.repo, issue))
            for comment in all_comments.get(issue["issue_number"], []):
                normalized_docs.append(
                    normalize_comment(
                        owner=settings.owner,
                        repo=settings.repo,
                        issue_number=issue["issue_number"],
                        comment=comment,
                        title=issue.get("issue_title", ""),
                    )
                )

    images_analyzed = 0
    image_artifacts_created = 0

    with timed_step(logger, "multimodal_image_processing", "Analyze capped images with Phi-3 Vision"):
        logger.info(
            "IMAGE_PROCESSING_LIMIT | accepted=%s | skipped=%s | configured_max=%s",
            accepted_image_urls,
            skipped_image_urls,
            settings.max_ocr_images,
        )
        for idx, (issue_number, title, image_url) in enumerate(image_records[: settings.max_ocr_images], start=1):
            suffix = image_url.split("?")[0].split(".")[-1].lower()
            ext = suffix if suffix in {"png", "jpg", "jpeg", "gif", "webp"} else "img"
            image_path = settings.raw_dir / "images" / f"issue_{issue_number}_{idx}.{ext}"
            if not download_image(image_url, image_path):
                continue

            mm_result = analyze_image_with_phi3(settings=settings, image_path=image_path, image_url=image_url)
            if bool(mm_result.get("success")):
                images_analyzed += 1

            ocr_text = str(mm_result.get("ocr_text", "")).strip()
            analysis_text = str(mm_result.get("analysis_text", "")).strip()

            if ocr_text:
                normalized_docs.append(
                    normalize_image_ocr(
                        owner=settings.owner,
                        repo=settings.repo,
                        issue_number=issue_number,
                        title=title,
                        image_url=image_url,
                        ocr_text=ocr_text,
                        source_path=str(image_path),
                    )
                )
                image_artifacts_created += 1

            if analysis_text:
                normalized_docs.append(
                    normalize_image_analysis(
                        owner=settings.owner,
                        repo=settings.repo,
                        issue_number=issue_number,
                        title=title,
                        image_url=image_url,
                        analysis_text=analysis_text,
                        source_path=str(image_path),
                    )
                )
                image_artifacts_created += 1

            logger.info(
                "IMAGE_PROCESSED | issue=%s | url=%s | path=%s | model=%s | phi3_success=%s | ocr_len=%s | analysis_len=%s",
                issue_number,
                image_url,
                image_path,
                mm_result.get("model_used", settings.phi3_vision_model),
                mm_result.get("success", False),
                len(ocr_text),
                len(analysis_text),
            )

    with timed_step(logger, "persist_raw_processed", "Persist raw and normalized artifacts to disk"):
        _save_json(settings.raw_dir / "issues.json", issues)
        _save_json(settings.raw_dir / "comments.json", all_comments)
        _save_json(settings.processed_dir / "normalized_docs.json", normalized_docs)

    with timed_step(logger, "chunk_docs", "Create chunks with metadata retention"):
        chunks = chunk_documents(normalized_docs)
        logger.info("CHUNKS_CREATED | count=%s", len(chunks))

    with timed_step(logger, "embed_and_index", "Generate embeddings and upsert into Chroma"):
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        logger.info("EMBED_PROGRESS | total_chunks=%s", len(chunk_texts))
        embeddings = embedder.embed_texts(chunk_texts)
        vector_count = vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)
        logger.info("CHROMA_UPSERT_DONE | vector_count=%s", vector_count)

    result = {
        "owner": settings.owner,
        "repo": settings.repo,
        "mcp_connected": client.mcp_connected,
        "discovered_tools": client.discovered_tools,
        "used_tools": client.used_tools,
        "rest_fallback_used": client.rest_fallback_used,
        "zero_result_mcp_call": client.zero_result_mcp_call,
        "issues_ingested": len(issues),
        "comments_ingested": total_comments,
        "image_urls_detected": total_image_urls,
        "supported_image_urls": accepted_image_urls,
        "skipped_image_urls": skipped_image_urls,
        "images_analyzed": images_analyzed,
        "image_artifacts_created": image_artifacts_created,
        "normalized_documents": len(normalized_docs),
        "chunks_created": len(chunks),
        "chroma_vector_count": vector_count,
    }
    logger.info("INGEST_SUMMARY | %s", result)
    return result

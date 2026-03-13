"""Answer synthesis using local Mistral endpoint with grounded evidence."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.ollama_client import OllamaClient, OllamaGenerationError

logger = get_logger("app.rag.answer_builder")


class AnswerBuilder:
    """Build grounded responses from retrieved chunks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.answer_diagnostics: list[dict[str, Any]] = []

    def build_answer(self, question: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
        evidence = self._to_evidence(retrieved)
        if not evidence:
            return {
                "answer": "Insufficient evidence found in indexed issues/comments.",
                "evidence": [],
                "issue_results": [],
                "llm_diagnostics": {"answer_generation": []},
            }

        if "similar issues" in question.lower() and "issue id" in question.lower():
            issue_results = self._build_issue_results(question=question, retrieved=retrieved)
            answer = f"Found {len(issue_results)} relevant issues discussing package/memory failures." if issue_results else "No relevant issues found from retrieved evidence."
            return {
                "answer": answer,
                "evidence": evidence,
                "issue_results": issue_results,
                "llm_diagnostics": {"answer_generation": self.answer_diagnostics},
            }

        compact_retrieved = []
        for item in retrieved[:5]:
            compact = dict(item)
            compact["chunk_text"] = (item.get("chunk_text", "") or "")[:520]
            compact_retrieved.append(compact)

        context = "\n\n".join(
            (
                f"[{idx + 1}] issue={item.get('issue_number')} source={item.get('source_type')} url={item.get('url')}\n"
                f"assignees={item.get('assignees', '')} assignee_names={item.get('assignee_names', '')} "
                f"comment_author={item.get('comment_author', '')} comment_author_name={item.get('comment_author_name', '')}\n"
                f"{item.get('chunk_text', '')}"
            )
            for idx, item in enumerate(compact_retrieved)
        )

        logger.info("ANSWER_CONTEXT | context_chars=%s | evidence_count=%s", len(context), len(evidence))

        prompt = (
            "You are a grounded assistant. Answer ONLY from evidence context. "
            "Do not invent people names, assignees, or commenters. "
            "If uncertain or absent, answer exactly: not clearly stated in the issue/comments.\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence Context:\n{context}\n\n"
            "Return a concise answer."
        )

        answer_text = self._call_mistral(prompt)
        return {
            "answer": answer_text,
            "evidence": evidence,
            "issue_results": [],
            "llm_diagnostics": {"answer_generation": self.answer_diagnostics},
        }

    def _build_issue_results(self, question: str, retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for item in retrieved:
            issue_number = item.get("issue_number")
            try:
                issue_number_int = int(issue_number)
            except Exception:
                continue
            grouped.setdefault(issue_number_int, []).append(item)

        results: list[dict[str, Any]] = []

        for issue_number, items in grouped.items():
            top = items[0]
            title = top.get("title") or ""
            url = top.get("url") or ""
            created_at = top.get("created_at")
            closed_at = top.get("closed_at")
            snippets = [((it.get("chunk_text") or "").strip().replace("\n", " "))[:320] for it in items[:3]]
            context = "\n".join(f"- {s}" for s in snippets if s)

            issue_prompt = (
                "Using ONLY this issue context, produce:\n"
                "1) concise summary\n"
                "2) likely resolution/fix/outcome\n"
                "Do not invent facts. If missing, say not clearly stated.\n\n"
                f"Issue #{issue_number}: {title}\n"
                f"Context:\n{context}\n"
            )
            llm_text = self._call_mistral(issue_prompt)

            summary, resolution = self._split_summary_resolution(llm_text)
            resolved_by = self._infer_resolved_by(" ".join(snippets))

            result = {
                "issue_id": issue_number,
                "title": title,
                "summary": summary,
                "resolution": resolution,
                "resolved_by": resolved_by,
                "open_duration": self._compute_open_duration(created_at, closed_at),
                "issue_url": url,
                "assignees": top.get("assignees", ""),
                "assignee_names": top.get("assignee_names", ""),
                "evidence": snippets,
            }
            results.append(result)

        return results

    def _call_mistral(self, prompt: str) -> str:
        client = OllamaClient(base_url=self.settings.mistral_base_url, default_max_retries=self.settings.ollama_max_retries)
        try:
            result = client.generate(
                component="answer_generation",
                model=self.settings.answer_model,
                prompt=prompt,
                connect_timeout_seconds=self.settings.answer_connect_timeout_seconds,
                first_byte_timeout_seconds=self.settings.answer_first_byte_timeout_seconds,
                total_timeout_seconds=self.settings.answer_timeout_seconds,
                max_retries=self.settings.ollama_max_retries,
                options={"temperature": 0.2},
                warmup_before_retry=True,
            )
            self.answer_diagnostics.append(result.diagnostics)
            return result.text or "Model returned empty response."
        except OllamaGenerationError as exc:
            self.answer_diagnostics.append(exc.diagnostics)
            logger.warning("ANSWER_MODEL_FALLBACK | reason=%s", exc.diagnostics.get("fallback_reason", str(exc)))
            return "Unable to call local Mistral endpoint; details not clearly stated from available evidence."

    @staticmethod
    def _split_summary_resolution(text: str) -> tuple[str, str]:
        lines = [line.strip(" -") for line in text.splitlines() if line.strip()]
        if not lines:
            return (
                "not clearly stated in the issue/comments",
                "not clearly stated in the issue/comments",
            )
        if len(lines) == 1:
            return lines[0], "not clearly stated in the issue/comments"
        return lines[0], " ".join(lines[1:])

    @staticmethod
    def _infer_resolved_by(text: str) -> str:
        patterns = [r"(?:fixed|resolved) by (@[A-Za-z0-9_-]+)", r"closed by (@[A-Za-z0-9_-]+)"]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return "not clearly stated in the issue/comments"

    @staticmethod
    def _compute_open_duration(created_at: str | None, closed_at: str | None) -> str:
        if not created_at or not closed_at:
            return "not clearly stated in the issue/comments"
        try:
            start = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
            delta = end - start
            days = delta.days
            hours = delta.seconds // 3600
            return f"{days}d {hours}h"
        except Exception:
            return "not clearly stated in the issue/comments"

    @staticmethod
    def _to_evidence(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for item in retrieved:
            snippet = (item.get("chunk_text") or "").strip().replace("\n", " ")
            evidence.append(
                {
                    "source_type": item.get("source_type", "issue"),
                    "issue_number": item.get("issue_number"),
                    "comment_id": item.get("comment_id"),
                    "title": item.get("title"),
                    "url": item.get("url", ""),
                    "assignees": item.get("assignees", ""),
                    "assignee_names": item.get("assignee_names", ""),
                    "comment_author": item.get("comment_author", ""),
                    "comment_author_name": item.get("comment_author_name", ""),
                    "snippet": snippet[:280],
                }
            )
        return evidence

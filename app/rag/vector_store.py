"""ChromaDB vector store wrapper with in-memory fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger("app.rag.vector_store")


@dataclass(slots=True)
class VectorStore:
    """Vector store abstraction for local persistence and retrieval."""

    settings: Settings
    collection_name: str = "tickets"
    _collection: Any = field(init=False, default=None)
    _fallback_docs: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            # Keep recruiter-demo logs clean.
            logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
            logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

            self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(self.settings.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = client.get_or_create_collection(name=self.collection_name)
            logger.info("CHROMA_READY | path=%s | collection=%s", self.settings.chroma_dir, self.collection_name)
        except Exception as exc:
            logger.warning("CHROMA_FALLBACK | reason=%s", exc)
            self._collection = None

    def upsert_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> int:
        """Upsert chunk embeddings and metadata, returning current vector count."""

        if not chunks:
            return self.count()

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["chunk_text"] for chunk in chunks]
        metadatas = [self._metadata_for_chunk(chunk) for chunk in chunks]

        if self._collection is not None:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            return self.count()

        for chunk, emb in zip(chunks, embeddings):
            item = dict(chunk)
            item["embedding"] = emb
            self._fallback_docs.append(item)
        return len(self._fallback_docs)

    def query(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """Retrieve top matching chunks by vector similarity."""

        if self._collection is not None:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            out: list[dict[str, Any]] = []
            for doc, meta, dist in zip(docs, metas, dists):
                row = dict(meta or {})
                row["chunk_text"] = doc
                row["distance"] = dist
                out.append(row)
            return out

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self._fallback_docs:
            score = self._cosine(query_embedding, item.get("embedding", []))
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)

        out = []
        for score, item in scored[:top_k]:
            row = dict(item)
            row["distance"] = 1.0 - score
            out.append(row)
        return out

    def count(self) -> int:
        """Get vector count from the active backend."""

        if self._collection is not None:
            return self._collection.count()
        return len(self._fallback_docs)

    @staticmethod
    def _metadata_for_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "source_type",
            "repo",
            "owner",
            "issue_number",
            "comment_id",
            "title",
            "url",
            "author",
            "assignees",
            "assignee_names",
            "comment_author",
            "comment_author_name",
            "created_at",
            "closed_at",
            "chunk_index",
        ]
        out: dict[str, Any] = {}
        for key in keys:
            value = chunk.get(key)
            if value is None:
                out[key] = ""
            elif isinstance(value, list):
                out[key] = ", ".join(str(x) for x in value if x is not None)
            else:
                out[key] = value
        return out

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return -1.0
        return dot / (na * nb)

"""Embedding service using sentence-transformers with deterministic fallback."""

from __future__ import annotations

import hashlib
from typing import Sequence

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger("app.rag.embedder")


class Embedder:
    """Generate embeddings for text chunks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = None

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.settings.embedding_model,
                cache_folder=self.settings.hf_cache_dir,
                local_files_only=self.settings.embedding_local_files_only,
            )
            logger.info(
                "EMBEDDER_READY | model=%s | local_files_only=%s | cache_dir=%s",
                self.settings.embedding_model,
                self.settings.embedding_local_files_only,
                self.settings.hf_cache_dir,
            )
        except Exception as exc:
            logger.warning("EMBEDDER_FALLBACK | reason=%s", exc)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts using sentence-transformers or deterministic fallback."""

        if not texts:
            return []

        if self.model is not None:
            vectors = self.model.encode(
                list(texts),
                batch_size=self.settings.embedding_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vectors.tolist()

        return [self._hash_embed(text) for text in texts]

    @staticmethod
    def _hash_embed(text: str, dims: int = 128) -> list[float]:
        """Deterministic low-cost fallback embedding."""

        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        values = []
        for i in range(dims):
            b = digest[i % len(digest)]
            values.append((b / 255.0) * 2.0 - 1.0)
        return values

"""Retriever orchestrating embedding query and vector search."""

from __future__ import annotations

from app.rag.embedder import Embedder
from app.rag.vector_store import VectorStore


class Retriever:
    """Retrieve top-k evidence chunks for a user query."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedder.embed_texts([query])[0]
        return self.vector_store.query(query_embedding=query_embedding, top_k=top_k)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from rag.vectorstore.chroma_store import ChromaVectorStore


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its Chroma distance score."""

    content: str
    metadata: dict[str, Any]
    score: float


class DocumentRetriever:
    """Retrieve relevant chunks from the configured Chroma vector store."""

    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        k: int = 4,
        metadata_filter: dict[str, Any] | None = None,
        max_distance: float | None = None,
    ) -> list[RetrievedChunk]:
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=metadata_filter,
        )
        retrieved = [
            RetrievedChunk(
                content=document.page_content,
                metadata=dict(document.metadata),
                score=float(score),
            )
            for document, score in results
        ]
        if max_distance is not None:
            retrieved = [item for item in retrieved if item.score <= max_distance]
        return retrieved

    def retrieve_context(
        self,
        query: str,
        k: int = 4,
        metadata_filter: dict[str, Any] | None = None,
        max_distance: float | None = None,
    ) -> str:
        chunks = self.retrieve(
            query=query,
            k=k,
            metadata_filter=metadata_filter,
            max_distance=max_distance,
        )
        return self.build_context(chunks)

    @staticmethod
    def build_context(chunks: list[RetrievedChunk]) -> str:
        context_parts: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source", "unknown source")
            title = chunk.metadata.get("title", "")
            heading = f"[{index}] {title} ({source})" if title else f"[{index}] {source}"
            context_parts.append(f"{heading}\n{chunk.content}")
        return "\n\n".join(context_parts)

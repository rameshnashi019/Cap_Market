from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Literal, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.config.settings import settings
from rag.data.chunker import DocumentChunk
from rag.embeddings.provider import EmbeddedChunk


class ChromaVectorStore:
    """Persistent LangChain Chroma store for embedded document chunks."""

    def __init__(
        self,
        embedding_function: Embeddings,
        persist_directory: str | Path | None = None,
        collection_name: str = "rag_documents",
        distance_metric: Literal["cosine", "l2", "ip"] = "cosine",
    ) -> None:
        if distance_metric not in {"cosine", "l2", "ip"}:
            raise ValueError("distance_metric must be 'cosine', 'l2', or 'ip'")
        self.persist_directory = Path(persist_directory or settings.vectorstore_dir)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.logger = logging.getLogger("rag.vectorstore.chroma")
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=str(self.persist_directory),
            collection_metadata={"hnsw:space": distance_metric},
        )

    @staticmethod
    def _chunk_id(chunk: DocumentChunk | EmbeddedChunk) -> str:
        source = str(chunk.metadata.get("source") or chunk.metadata.get("parent_file") or "")
        index = str(chunk.metadata.get("chunk_index", ""))
        content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:16]
        raw_id = f"{source}:{index}:{content_hash}"
        return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metadata.items()
            if value is not None and isinstance(value, (str, int, float, bool))
        }

    def upsert_embedded_chunks(self, chunks: Sequence[EmbeddedChunk]) -> list[str]:
        if not chunks:
            return []

        ids = [self._chunk_id(chunk) for chunk in chunks]
        self.store._collection.upsert(
            ids=ids,
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._clean_metadata(chunk.metadata) for chunk in chunks],
        )
        self.logger.info("Upserted %s embedded chunks into %s", len(chunks), self.collection_name)
        return ids

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embedding_function: Embeddings,
    ) -> list[str]:
        if not chunks:
            return []
        vectors = embedding_function.embed_documents([chunk.content for chunk in chunks])
        embedded_chunks = [
            EmbeddedChunk(
                content=chunk.content,
                embedding=[float(value) for value in vector],
                metadata=dict(chunk.metadata),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        return self.upsert_embedded_chunks(embedded_chunks)

    def delete_ids(self, ids: Sequence[str]) -> None:
        if ids:
            self.store.delete(ids=list(ids))
            self.logger.info("Deleted %s chunks from %s", len(ids), self.collection_name)

    def delete_source(self, source: str) -> None:
        self.store.delete(where={"source": source})
        self.logger.info("Deleted chunks for source %s", source)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        return self.store.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
        )

    def count(self) -> int:
        return int(self.store._collection.count())

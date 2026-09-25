"""Core retrieval workflow for a simple RAG application."""

from __future__ import annotations

import logging
from typing import Any

from rag.data.chunker import DocumentChunker
from rag.data.cleaner import DocumentCleaner
from rag.data.ingestion import DocumentIngestionManager
from rag.data.validator import QualityValidator
from rag.embeddings.provider import EmbeddingProvider
from rag.llm.generator import GeneratedAnswer, RAGAnswerGenerator
from rag.retrieval.retriever import DocumentRetriever
from rag.vectorstore.chroma_store import ChromaVectorStore


class RAGPipeline:
    """Coordinate ingestion, indexing, retrieval, and grounded generation."""

    def __init__(
        self,
        source_dir: str | None = None,
        ingestion_manager: DocumentIngestionManager | None = None,
        cleaner: DocumentCleaner | None = None,
        validator: QualityValidator | None = None,
        chunker: DocumentChunker | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: ChromaVectorStore | None = None,
        retriever: DocumentRetriever | None = None,
        answer_generator: RAGAnswerGenerator | None = None,
    ) -> None:
        self.logger = logging.getLogger("rag.retrieval.pipeline")
        self.documents: list[str] = []
        self.ingestion = ingestion_manager or DocumentIngestionManager(source_dir=source_dir)
        self.cleaner = cleaner or DocumentCleaner()
        self.validator = validator or QualityValidator()
        self.chunker = chunker or DocumentChunker()
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        self.vector_store = vector_store or ChromaVectorStore(
            embedding_function=self.embedding_provider
        )
        self.retriever = retriever or DocumentRetriever(self.vector_store)
        self.answer_generator = answer_generator or RAGAnswerGenerator(self.retriever)

    def sync_and_index(self) -> dict[str, Any]:
        """Process changed source files and synchronize their chunks to Chroma."""
        changes = self.ingestion.sync_documents()
        if not any(changes.values()) and self.vector_store.count() == 0:
            changes["added"] = self.ingestion.list_documents()
        indexed: list[str] = []
        skipped: list[dict[str, Any]] = []

        for file_path in changes["removed"]:
            self.vector_store.delete_source(str(file_path.resolve()))

        for file_path in changes["added"] + changes["updated"]:
            parsed = self.ingestion.parse_document(file_path)
            if parsed.status != "success":
                skipped.append({"path": parsed.file_path, "reason": parsed.error or "parse failed"})
                continue

            cleaned = self.cleaner.clean(parsed)
            validation = self.validator.validate(cleaned)
            if not validation["is_valid"]:
                skipped.append(
                    {
                        "path": parsed.file_path,
                        "reason": "quality validation failed",
                        "checks": validation["checks"],
                    }
                )
                continue

            chunks = self.chunker.chunk(cleaned)
            if not chunks:
                skipped.append({"path": parsed.file_path, "reason": "no chunks produced"})
                continue

            if file_path in changes["updated"]:
                self.vector_store.delete_source(parsed.file_path)
            self.vector_store.upsert_chunks(chunks, self.embedding_provider)
            indexed.append(parsed.file_path)

        result = {
            "added": changes["added"],
            "updated": changes["updated"],
            "removed": changes["removed"],
            "indexed": indexed,
            "skipped": skipped,
        }
        self.logger.info(
            "RAG sync complete: indexed=%s skipped=%s removed=%s",
            len(indexed),
            len(skipped),
            len(changes["removed"]),
        )
        return result

    def ask(self, question: str, k: int = 4) -> GeneratedAnswer:
        """Retrieve evidence and return a grounded answer."""
        return self.answer_generator.generate(question, k=k)

    def add_documents(self, documents: list[str]) -> None:
        """Keep the original in-memory API for compatibility with early callers."""
        self.documents.extend(documents)

    def query(self, question: str) -> str:
        if not self.documents:
            return "No documents available yet. Add data first."
        return f"Query: {question} | Relevant docs: {len(self.documents)}"

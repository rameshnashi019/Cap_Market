from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.data.ingestion import CommonDocument

ChunkStrategy = Literal["recursive", "paragraph", "sentence", "fixed", "markdown"]


@dataclass
class DocumentChunk:
    """A retrievable piece of a normalized and validated document."""

    content: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """Split documents using interchangeable strategies.

    Recursive splitting is the default because it preserves paragraphs and
    sentences as long as possible before falling back to smaller boundaries.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _recursive_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            strip_whitespace=True,
        )

    def _split_sentences(self, content: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", content.strip())
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _sentence_chunks(self, content: str) -> list[str]:
        sentences = self._split_sentences(content)
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > self.chunk_size:
                chunks.append(current)
                overlap = current[-self.chunk_overlap :].strip()
                current = f"{overlap} {sentence}".strip()
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _markdown_chunks(self, content: str) -> list[str]:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "header_1"), ("##", "header_2"), ("###", "header_3")],
            strip_headers=False,
        )
        sections = splitter.split_text(content)
        chunks: list[str] = []
        recursive = self._recursive_splitter()
        for section in sections:
            section_text = section.page_content.strip()
            if section_text:
                chunks.extend(recursive.split_text(section_text))
        return chunks

    def _split_content(self, content: str, strategy: ChunkStrategy) -> list[str]:
        if strategy == "recursive":
            return self._recursive_splitter().split_text(content)
        if strategy == "paragraph":
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
            recursive = self._recursive_splitter()
            return [chunk for paragraph in paragraphs for chunk in recursive.split_text(paragraph)]
        if strategy == "sentence":
            return self._sentence_chunks(content)
        if strategy == "fixed":
            splitter = CharacterTextSplitter(
                separator="",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                strip_whitespace=True,
            )
            return splitter.split_text(content)
        if strategy == "markdown":
            return self._markdown_chunks(content)
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    def chunk(self, document: CommonDocument, strategy: ChunkStrategy = "recursive") -> list[DocumentChunk]:
        if document.status != "success":
            return []
        if not document.content.strip():
            return []

        contents = self._split_content(document.content, strategy)
        chunk_count = len(contents)
        chunks: list[DocumentChunk] = []
        for index, content in enumerate(contents):
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "chunk_index": index,
                    "chunk_count": chunk_count,
                    "chunk_strategy": strategy,
                    "chunk_size": len(content),
                    "parent_file": document.file_path,
                }
            )
            chunks.append(DocumentChunk(content=content, chunk_index=index, metadata=metadata))
        return chunks

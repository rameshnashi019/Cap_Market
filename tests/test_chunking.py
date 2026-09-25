from pathlib import Path

import pytest

from rag.data.chunk_evaluator import ChunkEvaluator
from rag.data.chunker import DocumentChunker
from rag.data.ingestion import DocumentIngestionManager


@pytest.fixture
def document(tmp_path: Path):
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    file_path = source_dir / "guide.md"
    file_path.write_text(
        "# Getting Started\n\n"
        "This is the first paragraph with useful retrieval content. "
        "It has several sentences for sentence-aware splitting.\n\n"
        "## Configuration\n\n"
        "Configure the application carefully and keep the source context available.",
        encoding="utf-8",
    )
    return DocumentIngestionManager(source_dir=source_dir).parse_document(file_path)


def test_chunker_supports_multiple_strategies(document) -> None:
    chunker = DocumentChunker(chunk_size=70, chunk_overlap=10)

    for strategy in ("recursive", "paragraph", "sentence", "fixed", "markdown"):
        chunks = chunker.chunk(document, strategy=strategy)

        assert chunks
        assert all(chunk.content for chunk in chunks)
        assert all(chunk.metadata["chunk_strategy"] == strategy for chunk in chunks)
        assert chunks[0].metadata["parent_file"] == document.file_path


def test_recursive_chunking_preserves_metadata_and_evaluates_well(document) -> None:
    chunker = DocumentChunker(chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk(document)

    result = ChunkEvaluator().evaluate(
        chunks,
        source_text=document.content,
        max_chunk_size=120,
    )

    assert result["is_valid"] is True
    assert result["checks"]["no_empty_chunks"] is True
    assert result["checks"]["no_duplicate_chunks"] is True
    assert result["checks"]["source_coverage"] is True
    assert all(chunk.metadata["chunk_count"] == len(chunks) for chunk in chunks)


def test_chunk_evaluator_rejects_duplicate_chunks(document) -> None:
    chunks = DocumentChunker(chunk_size=120, chunk_overlap=20).chunk(document)
    duplicate_chunks = chunks + [chunks[0]]

    result = ChunkEvaluator().evaluate(duplicate_chunks)

    assert result["is_valid"] is False
    assert result["duplicate_chunk_count"] == 1
    assert result["checks"]["no_duplicate_chunks"] is False

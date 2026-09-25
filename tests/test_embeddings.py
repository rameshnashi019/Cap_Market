from typing import Any

import pytest

from rag.data.chunker import DocumentChunk
from rag.embeddings.provider import EmbeddingProvider


class FakeEmbeddingModel:
    def encode(self, texts: list[str], **_: Any) -> list[list[float]]:
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]

    def get_sentence_embedding_dimension(self) -> int:
        return 2


class FailingEmbeddingModel:
    def encode(self, texts: list[str], **_: Any) -> list[list[float]]:
        raise RuntimeError("primary provider unavailable")


class FallbackEmbeddingModel:
    def encode(self, texts: list[str], **_: Any) -> list[list[float]]:
        return [[99.0, float(index)] for index, _ in enumerate(texts)]


def test_embedding_provider_embeds_texts_and_queries() -> None:
    provider = EmbeddingProvider(model=FakeEmbeddingModel())

    embeddings = provider.embed_texts(["first text", "second text"])
    query_embedding = provider.embed_query("search text")

    assert embeddings == [[10.0, 0.0], [11.0, 1.0]]
    assert query_embedding == [11.0, 0.0]
    assert provider.dimension == 2


def test_embedding_provider_embeds_chunks_and_preserves_metadata() -> None:
    chunks = [
        DocumentChunk(
            content="Useful content",
            chunk_index=0,
            metadata={"source": "guide.md", "chunk_strategy": "recursive"},
        )
    ]

    embedded = EmbeddingProvider(model=FakeEmbeddingModel()).embed_chunks(chunks)

    assert embedded[0].content == "Useful content"
    assert embedded[0].embedding == [14.0, 0.0]
    assert embedded[0].metadata == chunks[0].metadata


def test_embedding_provider_rejects_empty_text() -> None:
    provider = EmbeddingProvider(model=FakeEmbeddingModel())

    with pytest.raises(ValueError, match="empty text"):
        provider.embed_texts([" "])


def test_embedding_provider_falls_back_when_primary_fails() -> None:
    provider = EmbeddingProvider(
        model=FailingEmbeddingModel(),
        fallback_provider="openai",
        fallback_model=FallbackEmbeddingModel(),
    )

    embeddings = provider.embed_texts(["document text"])
    query_embedding = provider.embed_query("search text")

    assert embeddings == [[99.0, 0.0]]
    assert query_embedding == [99.0, 0.0]
    assert provider._fallback_used is True

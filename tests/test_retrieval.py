from pathlib import Path

from langchain_core.embeddings import Embeddings

from rag.data.chunker import DocumentChunk
from rag.retrieval.retriever import DocumentRetriever
from rag.vectorstore.chroma_store import ChromaVectorStore


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def test_retriever_returns_filtered_chunks_and_context(tmp_path: Path) -> None:
    embedding = FakeEmbeddings()
    store = ChromaVectorStore(
        embedding_function=embedding,
        persist_directory=tmp_path / "chroma",
        collection_name="test_retrieval",
    )
    store.upsert_chunks(
        [
            DocumentChunk(
                content="Authentication uses secure tokens.",
                chunk_index=0,
                metadata={"source": "security.md", "title": "Security"},
            ),
            DocumentChunk(
                content="Deployment uses a release pipeline.",
                chunk_index=1,
                metadata={"source": "deploy.md", "title": "Deployment"},
            ),
        ],
        embedding,
    )

    retriever = DocumentRetriever(store)
    results = retriever.retrieve(
        "security",
        k=2,
        metadata_filter={"source": "security.md"},
    )
    context = retriever.retrieve_context(
        "security",
        k=2,
        metadata_filter={"source": "security.md"},
    )

    assert len(results) == 1
    assert results[0].metadata["title"] == "Security"
    assert results[0].score >= 0
    assert "Authentication uses secure tokens." in context
    assert "security.md" in context


def test_retriever_applies_distance_threshold(tmp_path: Path) -> None:
    embedding = FakeEmbeddings()
    store = ChromaVectorStore(
        embedding_function=embedding,
        persist_directory=tmp_path / "chroma",
        collection_name="test_threshold",
    )
    store.upsert_chunks(
        [
            DocumentChunk(
                content="A document with searchable content.",
                chunk_index=0,
                metadata={"source": "notes.txt"},
            )
        ],
        embedding,
    )

    results = DocumentRetriever(store).retrieve("different", max_distance=0.0)

    assert results == []

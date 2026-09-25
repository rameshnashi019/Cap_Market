from pathlib import Path

from langchain_core.embeddings import Embeddings

from rag.data.chunker import DocumentChunk
from rag.vectorstore.chroma_store import ChromaVectorStore


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def test_chroma_upserts_searches_and_deletes_source(tmp_path: Path) -> None:
    embedding = FakeEmbeddings()
    store = ChromaVectorStore(
        embedding_function=embedding,
        persist_directory=tmp_path / "chroma",
        collection_name="test_documents",
    )
    chunks = [
        DocumentChunk(
            content="A useful document chunk",
            chunk_index=0,
            metadata={"source": "guide.txt", "title": "Guide"},
        )
    ]

    ids = store.upsert_chunks(chunks, embedding)
    results = store.similarity_search("useful", k=1)

    assert len(ids) == 1
    assert store.count() == 1
    assert store.store._collection.metadata["hnsw:space"] == "cosine"
    assert len(results) == 1
    assert results[0][0].page_content == chunks[0].content
    assert results[0][0].metadata["source"] == "guide.txt"

    store.delete_source("guide.txt")

    assert store.count() == 0
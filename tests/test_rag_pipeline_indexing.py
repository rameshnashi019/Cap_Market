from pathlib import Path

from langchain_core.embeddings import Embeddings

from rag.data.ingestion import DocumentIngestionManager
from rag.llm.generator import GeneratedAnswer
from rag.retrieval.rag_pipeline import RAGPipeline
from rag.vectorstore.chroma_store import ChromaVectorStore


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


class FakeAnswerGenerator:
    def generate(self, question: str, k: int = 4) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=f"Answer for: {question}",
            queries=[question],
            sources=["guide.txt"],
            retrieved_chunks=k,
        )


def test_pipeline_indexes_changes_and_answers(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    file_path = source_dir / "guide.txt"
    file_path.write_text(
        "Authentication uses secure tokens and requires a configured expiration policy.",
        encoding="utf-8",
    )

    embedding = FakeEmbeddings()
    vector_store = ChromaVectorStore(
        embedding_function=embedding,
        persist_directory=tmp_path / "chroma",
        collection_name="pipeline_test",
    )
    pipeline = RAGPipeline(
        ingestion_manager=DocumentIngestionManager(source_dir=source_dir),
        embedding_provider=embedding,  # type: ignore[arg-type]
        vector_store=vector_store,
        answer_generator=FakeAnswerGenerator(),  # type: ignore[arg-type]
    )

    sync_result = pipeline.sync_and_index()
    answer = pipeline.ask("How does authentication work?", k=2)

    assert len(sync_result["indexed"]) == 1
    assert sync_result["skipped"] == []
    assert vector_store.count() > 0
    assert answer.answer == "Answer for: How does authentication work?"
    assert answer.sources == ["guide.txt"]

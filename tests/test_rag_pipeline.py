from rag.retrieval.rag_pipeline import RAGPipeline


def test_rag_pipeline_returns_message_when_empty() -> None:
    pipeline = RAGPipeline()
    assert "No documents available yet" in pipeline.query("What is RAG?")


def test_rag_pipeline_counts_documents() -> None:
    pipeline = RAGPipeline()
    pipeline.add_documents(["doc1", "doc2"])
    result = pipeline.query("What is RAG?")
    assert "Relevant docs: 2" in result

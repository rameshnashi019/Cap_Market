from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from rag.api.routes import create_app
from rag.data.ingestion import DocumentIngestionManager
from rag.embeddings.provider import EmbeddingProvider
from rag.retrieval.rag_pipeline import RAGPipeline
from rag.vectorstore.chroma_store import ChromaVectorStore


@dataclass
class FakeAnswer:
    answer: str
    queries: list[str]
    sources: list[str]
    retrieved_chunks: int


class FakePipeline:
    def ask(self, question: str, k: int = 4) -> FakeAnswer:
        return FakeAnswer(
            answer=f"Grounded answer for {question}",
            queries=[question],
            sources=["guide.md"],
            retrieved_chunks=k,
        )


def make_client() -> TestClient:
    app = create_app(
        pipeline=FakePipeline(),  # type: ignore[arg-type]
        config={
            "secret_key": "test-secret",
            "app_username": "test-user",
            "app_password": "test-password",
        },
    )
    return TestClient(app)


def test_login_is_homepage_and_chat_requires_authentication() -> None:
    client = make_client()

    home = client.get("/")
    protected = client.get("/chat", follow_redirects=False)
    api = client.post("/api/chat", json={"question": "hello"})

    assert home.status_code == 200
    assert "Sign in to continue" in home.text
    assert protected.status_code == 303
    assert protected.headers["location"] == "/"
    assert api.status_code == 401


def test_authenticated_user_can_open_chat_and_ask_question() -> None:
    client = make_client()

    login = client.post(
        "/login",
        data={"username": "test-user", "password": "test-password"},
        follow_redirects=False,
    )
    chat = client.get("/chat")
    response = client.post("/api/chat", json={"question": "What is in the guide?", "k": 2})

    assert login.status_code == 303
    assert login.headers["location"] == "/chat"
    assert chat.status_code == 200
    assert "Document" in chat.text
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Grounded answer for What is in the guide?",
        "queries": ["What is in the guide?"],
        "retrieved_chunks": 2,
    }


def test_invalid_login_is_rejected() -> None:
    client = make_client()

    response = client.post(
        "/login",
        data={"username": "wrong", "password": "wrong"},
    )

    assert response.status_code == 401
    assert "Invalid username or password" in response.text


def test_create_app_indexes_documents_on_startup(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    (source_dir / "guide.txt").write_text(
        "Trade capture is the first stage. Reconciliation compares records after settlement.",
        encoding="utf-8",
    )

    embedding = EmbeddingProvider(provider="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = ChromaVectorStore(
        embedding_function=embedding,
        persist_directory=tmp_path / "chroma",
        collection_name="startup_index_test",
    )
    pipeline = RAGPipeline(
        source_dir=source_dir,
        ingestion_manager=DocumentIngestionManager(source_dir=source_dir),
        embedding_provider=embedding,
        vector_store=vector_store,
    )

    app = create_app(pipeline=pipeline)

    assert app.state.rag_pipeline.vector_store.count() > 0

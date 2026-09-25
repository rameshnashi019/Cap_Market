from dataclasses import dataclass
from typing import Any

from rag.llm.generator import GeneratedAnswer, QueryDecomposer, RAGAnswerGenerator
from rag.retrieval.retriever import RetrievedChunk


@dataclass
class FakeResponse:
    content: str


class FakeDecompositionLLM:
    def invoke(self, _: Any) -> FakeResponse:
        return FakeResponse("- authentication requirements\n- token expiration\n- authentication requirements")


class FakeAnswerLLM:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, prompt: Any) -> FakeResponse:
        self.calls += 1
        rendered = "\n".join(message.content for message in prompt.messages)
        assert "using only the supplied context" in rendered
        assert "[1]" in rendered
        return FakeResponse("Authentication uses secure tokens [1].")


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        assert query
        assert k > 0
        return self.chunks

    @staticmethod
    def build_context(chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            f"[{index}] {chunk.metadata['source']}\n{chunk.content}"
            for index, chunk in enumerate(chunks, start=1)
        )


def test_query_decomposer_returns_unique_bounded_queries() -> None:
    decomposer = QueryDecomposer(llm=FakeDecompositionLLM(), max_queries=3)

    queries = decomposer.decompose("How does authentication work?")

    assert queries == ["authentication requirements", "token expiration"]


def test_generator_uses_retrieved_context_and_tracks_sources() -> None:
    answer_llm = FakeAnswerLLM()
    retriever = FakeRetriever(
        [
            RetrievedChunk(
                content="Authentication uses secure tokens.",
                metadata={"source": "security.md"},
                score=0.1,
            )
        ]
    )
    generator = RAGAnswerGenerator(
        retriever=retriever,  # type: ignore[arg-type]
        llm=answer_llm,
        decomposer=QueryDecomposer(llm=FakeDecompositionLLM()),
    )

    result = generator.generate("How does authentication work?", k=2)

    assert isinstance(result, GeneratedAnswer)
    assert result.answer == "Authentication uses secure tokens."
    assert result.sources == ["security.md"]
    assert result.retrieved_chunks == 1
    assert answer_llm.calls == 1


def test_generator_returns_safe_answer_without_context() -> None:
    answer_llm = FakeAnswerLLM()
    generator = RAGAnswerGenerator(
        retriever=FakeRetriever([]),  # type: ignore[arg-type]
        llm=answer_llm,
        decomposer=QueryDecomposer(llm=FakeDecompositionLLM()),
    )

    result = generator.generate("What is unavailable?")

    assert "not have enough information" in result.answer
    assert result.sources == []
    assert answer_llm.calls == 0


def test_generator_strips_citation_markers_from_final_answer() -> None:
    generator = RAGAnswerGenerator(
        retriever=FakeRetriever(
            [
                RetrievedChunk(
                    content="Authentication uses secure tokens.",
                    metadata={"source": "security.md"},
                    score=0.1,
                )
            ]
        ),
        llm=FakeAnswerLLM(),
        decomposer=QueryDecomposer(llm=FakeDecompositionLLM()),
    )

    result = generator.generate("How does authentication work?", k=2)

    assert "[1]" not in result.answer
    assert "security.md" not in result.answer

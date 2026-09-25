from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag.config.settings import settings
from rag.llm.prompts import ANSWER_PROMPT, DECOMPOSITION_PROMPT
from rag.retrieval.retriever import DocumentRetriever, RetrievedChunk


@dataclass
class GeneratedAnswer:
    answer: str
    queries: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    retrieved_chunks: int = 0


class QueryDecomposer:
    """Turn a complex user question into focused retrieval queries."""

    def __init__(self, llm: Any | None = None, max_queries: int = 4) -> None:
        if max_queries <= 0:
            raise ValueError("max_queries must be greater than zero")
        load_dotenv()
        self.llm = llm
        self.max_queries = max_queries

    def _load_llm(self) -> Any:
        if self.llm is None:
            self.llm = ChatOpenAI(model=settings.openai_model, temperature=0)
        return self.llm

    def decompose(self, question: str) -> list[str]:
        if not question.strip():
            raise ValueError("question must not be empty")

        prompt = DECOMPOSITION_PROMPT.invoke({"question": question.strip()})
        response = self._load_llm().invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        queries = []
        for line in text.splitlines():
            cleaned = line.strip().lstrip("-* ").strip()
            if cleaned and cleaned not in queries:
                queries.append(cleaned)
        return queries[: self.max_queries] or [question.strip()]


class RAGAnswerGenerator:
    """Retrieve evidence and generate a grounded answer from it."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm: Any | None = None,
        decomposer: QueryDecomposer | None = None,
    ) -> None:
        load_dotenv()
        self.retriever = retriever
        self.llm = llm
        self.decomposer = decomposer or QueryDecomposer(llm=llm)

    def _load_llm(self) -> Any:
        if self.llm is None:
            self.llm = ChatOpenAI(model=settings.openai_model, temperature=0)
        return self.llm

    @staticmethod
    def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen: set[tuple[str, str]] = set()
        unique: list[RetrievedChunk] = []
        for chunk in chunks:
            key = (chunk.metadata.get("source", ""), chunk.content)
            if key not in seen:
                seen.add(key)
                unique.append(chunk)
        return unique

    @staticmethod
    def _sources(chunks: list[RetrievedChunk]) -> list[str]:
        return list(dict.fromkeys(
            str(chunk.metadata["source"])
            for chunk in chunks
            if chunk.metadata.get("source")
        ))

    def generate(self, question: str, k: int = 4) -> GeneratedAnswer:
        if not question.strip():
            raise ValueError("question must not be empty")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        queries = self.decomposer.decompose(question)
        chunks: list[RetrievedChunk] = []
        for query in queries:
            chunks.extend(self.retriever.retrieve(query, k=k))
        chunks = self._deduplicate(chunks)

        if not chunks:
            return GeneratedAnswer(
                answer="I do not have enough information in the documents to answer that question.",
                queries=queries,
            )

        context = self.retriever.build_context(chunks)
        prompt = ANSWER_PROMPT.invoke({"question": question.strip(), "context": context})
        response = self._load_llm().invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        cleaned_answer = answer.strip()
        cleaned_answer = __import__("re").sub(r"\s*\[[0-9]+\]\s*", " ", cleaned_answer)
        cleaned_answer = __import__("re").sub(r"\s+(?=[.,;:!?])", "", cleaned_answer)
        cleaned_answer = __import__("re").sub(r"\s+", " ", cleaned_answer).strip()
        return GeneratedAnswer(
            answer=cleaned_answer,
            queries=queries,
            sources=self._sources(chunks),
            retrieved_chunks=len(chunks),
        )

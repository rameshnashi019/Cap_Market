from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from rag.config.settings import settings
from rag.data.chunker import DocumentChunk


@dataclass
class EmbeddedChunk:
    """A chunk and its embedding, ready for vector-store insertion."""

    content: str
    embedding: list[float]
    metadata: dict[str, Any]


class EmbeddingProvider(Embeddings):
    """Generate embeddings through a LangChain embedding implementation."""

    def __init__(
        self,
        provider: str = "huggingface",
        model_name: str | None = None,
        model: Any | None = None,
        normalize_embeddings: bool = True,
        fallback_provider: str | None = None,
        fallback_model_name: str | None = None,
        fallback_model: Any | None = None,
    ) -> None:
        load_dotenv()
        self._validate_provider(provider)
        if fallback_provider is not None:
            self._validate_provider(fallback_provider)
            if fallback_provider == provider:
                raise ValueError("fallback_provider must differ from provider")

        self.provider = provider
        self.model_name = model_name or (
            settings.embedding_model if provider == "huggingface" else "text-embedding-3-small"
        )
        self._model = model
        self.normalize_embeddings = normalize_embeddings
        self.fallback_provider = fallback_provider or (
            "huggingface" if provider == "openai" else None
        )
        self.fallback_model_name = fallback_model_name
        self._fallback_model = fallback_model
        self._fallback_used = False
        self.logger = logging.getLogger("rag.embeddings")

    @staticmethod
    def _validate_provider(provider: str) -> None:
        if provider not in {"huggingface", "openai"}:
            raise ValueError("provider must be 'huggingface' or 'openai'")

    def _create_model(self, provider: str, model_name: str | None) -> Any:
        selected_name = model_name or (
            settings.embedding_model if provider == "huggingface" else "text-embedding-3-small"
        )
        if provider == "huggingface":
            return HuggingFaceEmbeddings(
                    model_name=selected_name,
                    encode_kwargs={"normalize_embeddings": self.normalize_embeddings},
                )
        return OpenAIEmbeddings(model=selected_name)

    def _load_model(self) -> Any:
        if self._model is None:
            self._model = self._create_model(self.provider, self.model_name)
        return self._model

    def _load_fallback_model(self) -> Any:
        if self.fallback_provider is None:
            raise RuntimeError("No fallback embedding provider is configured")
        if self._fallback_model is None:
            self._fallback_model = self._create_model(
                self.fallback_provider,
                self.fallback_model_name,
            )
        return self._fallback_model

    def _activate_fallback(self, error: Exception) -> Any:
        if self._fallback_used:
            return self._load_fallback_model()
        if self.fallback_provider is None:
            raise error
        self._fallback_used = True
        self.logger.warning(
            "Embedding provider '%s' failed; falling back to '%s': %s",
            self.provider,
            self.fallback_provider,
            error,
        )
        return self._load_fallback_model()

    def _embed_documents_with_model(self, model: Any, texts: list[str]) -> Any:
        if hasattr(model, "embed_documents"):
            return model.embed_documents(texts)
        return model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

    @staticmethod
    def _to_float_lists(vectors: Any) -> list[list[float]]:
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return [[float(value) for value in vector] for vector in vectors]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")
        if not texts:
            return []

        try:
            vectors = self._embed_documents_with_model(self._load_model(), list(texts))
        except Exception as primary_error:
            try:
                fallback = self._activate_fallback(primary_error)
                vectors = self._embed_documents_with_model(fallback, list(texts))
            except Exception as fallback_error:
                raise RuntimeError("All embedding providers failed") from fallback_error
        return self._to_float_lists(vectors)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Implement LangChain's standard document-embedding interface."""
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("Cannot embed empty text")
        try:
            model = self._load_model()
            if hasattr(model, "embed_query"):
                return [float(value) for value in model.embed_query(query)]
            return self._to_float_lists(
                self._embed_documents_with_model(model, [query])
            )[0]
        except Exception as primary_error:
            try:
                fallback = self._activate_fallback(primary_error)
                if hasattr(fallback, "embed_query"):
                    return [float(value) for value in fallback.embed_query(query)]
                return self._to_float_lists(
                    self._embed_documents_with_model(fallback, [query])
                )[0]
            except Exception as fallback_error:
                raise RuntimeError("All embedding providers failed") from fallback_error

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> list[EmbeddedChunk]:
        contents = [chunk.content for chunk in chunks]
        vectors = self.embed_texts(contents)
        return [
            EmbeddedChunk(
                content=chunk.content,
                embedding=vector,
                metadata=dict(chunk.metadata),
            )
            for chunk, vector in zip(chunks, vectors)
        ]

    @property
    def dimension(self) -> int:
        model = self._load_model()
        if hasattr(model, "embedding_dimension"):
            dimension = model.embedding_dimension
        elif hasattr(model, "client") and hasattr(model.client, "get_sentence_embedding_dimension"):
            dimension = model.client.get_sentence_embedding_dimension()
        elif hasattr(model, "get_sentence_embedding_dimension"):
            dimension = model.get_sentence_embedding_dimension()
        else:
            dimension = len(self.embed_query("dimension probe"))
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Embedding model returned an invalid dimension")
        return dimension

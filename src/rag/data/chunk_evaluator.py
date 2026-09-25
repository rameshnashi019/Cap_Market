from __future__ import annotations

import re
from typing import Any, Sequence

from rag.data.chunker import DocumentChunk


class ChunkEvaluator:
    """Evaluate whether generated chunks are usable for retrieval."""

    def evaluate(
        self,
        chunks: Sequence[DocumentChunk],
        source_text: str | None = None,
        max_chunk_size: int | None = None,
    ) -> dict[str, Any]:
        contents = [chunk.content.strip() for chunk in chunks]
        non_empty = [content for content in contents if content]
        normalized = [re.sub(r"\s+", " ", content).lower() for content in non_empty]
        duplicate_count = len(normalized) - len(set(normalized))
        oversized_count = (
            sum(len(content) > max_chunk_size for content in non_empty)
            if max_chunk_size is not None
            else 0
        )
        source_coverage = self._source_coverage(source_text, non_empty)

        checks = {
            "has_chunks": bool(non_empty),
            "no_empty_chunks": len(non_empty) == len(contents),
            "no_duplicate_chunks": duplicate_count == 0,
            "within_max_size": oversized_count == 0,
            "source_coverage": source_coverage >= 0.95 if source_text else True,
        }
        passed_checks = sum(checks.values())
        score = passed_checks / len(checks)

        return {
            "is_valid": all(checks.values()),
            "quality_score": round(score, 3),
            "checks": checks,
            "chunk_count": len(chunks),
            "empty_chunk_count": len(contents) - len(non_empty),
            "duplicate_chunk_count": duplicate_count,
            "oversized_chunk_count": oversized_count,
            "min_chunk_size": min((len(content) for content in non_empty), default=0),
            "max_chunk_size": max((len(content) for content in non_empty), default=0),
            "average_chunk_size": round(
                sum(len(content) for content in non_empty) / len(non_empty), 2
            )
            if non_empty
            else 0,
            "source_coverage": round(source_coverage, 3),
        }

    @staticmethod
    def _source_coverage(source_text: str | None, chunks: list[str]) -> float:
        if not source_text:
            return 0.0
        source_words = set(re.findall(r"\w+", source_text.lower()))
        chunk_words = set(re.findall(r"\w+", " ".join(chunks).lower()))
        if not source_words:
            return 1.0
        return len(source_words & chunk_words) / len(source_words)

from __future__ import annotations

import re
from typing import Any

from rag.data.ingestion import CommonDocument


class QualityValidator:
    """Validate cleaned documents before chunking and embedding."""

    @staticmethod
    def _has_empty_text(document: CommonDocument) -> bool:
        return not document.content or not document.content.strip()

    @staticmethod
    def _min_max_length(document: CommonDocument, min_chars: int = 20, max_chars: int = 200000) -> bool:
        text_length = len(document.content.strip())
        return min_chars <= text_length <= max_chars

    @staticmethod
    def _has_duplicate_text(document: CommonDocument) -> bool:
        if not document.content:
            return False
        words = re.findall(r"\b\w+\b", document.content.lower())
        return len(set(words)) < len(words) * 0.5

    @staticmethod
    def _has_garbage_characters(document: CommonDocument) -> bool:
        garbage_pattern = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]+")
        return bool(garbage_pattern.search(document.content))

    @staticmethod
    def _language_quality(document: CommonDocument) -> bool:
        if not document.content.strip():
            return False
        alphabetic = sum(ch.isalpha() for ch in document.content)
        total = len(document.content)
        if total == 0:
            return False
        return (alphabetic / total) > 0.2

    @staticmethod
    def _ocr_quality(document: CommonDocument) -> bool:
        if not document.content:
            return False
        suspicious = [
            "@@@@",
            "####",
            "$$$$",
            "l l l",
            "????",
            "zzzz",
        ]
        return not any(token in document.content.lower() for token in suspicious)

    def validate(self, document: CommonDocument) -> dict[str, Any]:
        result = {
            "is_valid": True,
            "checks": {},
        }

        checks = {
            "empty_text": not self._has_empty_text(document),
            "min_max_length": self._min_max_length(document),
            "duplicate_text": not self._has_duplicate_text(document),
            "garbage_characters": not self._has_garbage_characters(document),
            "language_quality": self._language_quality(document),
            "ocr_quality": self._ocr_quality(document),
        }

        result["checks"] = checks
        result["is_valid"] = all(checks.values())

        if not result["is_valid"]:
            document.metadata["validation_failed"] = [
                name for name, passed in checks.items() if not passed
            ]
        else:
            document.metadata["validation_failed"] = []

        document.metadata["validated"] = result["is_valid"]
        return result

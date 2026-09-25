from __future__ import annotations

import re
from typing import Any

from rag.data.ingestion import CommonDocument


class DocumentCleaner:
    """Clean normalized document text before chunking and indexing."""

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" ?\n ?", "\n", text)
        return text.strip()

    @staticmethod
    def _remove_empty_documents(text: str) -> str:
        if not text or not text.strip():
            return ""
        return text

    @staticmethod
    def _remove_page_numbers(text: str) -> str:
        return re.sub(r"(?m)^(?:Page\s*\d+|\d+)$", "", text).strip()

    @staticmethod
    def _remove_duplicate_lines(text: str) -> str:
        seen: set[str] = set()
        result: list[str] = []
        for line in text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if cleaned.lower() in seen:
                continue
            seen.add(cleaned.lower())
            result.append(line)
        return "\n".join(result)

    @staticmethod
    def _remove_boilerplate(text: str) -> str:
        patterns = [
            r"(?i)^.*?confidential.*$",
            r"(?i)^.*?copyright.*$",
            r"(?i)^.*?all rights reserved.*$",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        return text

    @staticmethod
    def _fix_broken_sentences(text: str) -> str:
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return re.sub(r"(?<!\.)\n(?=[A-Z])", " ", text)

    @staticmethod
    def _normalize_tables(text: str) -> str:
        text = re.sub(r"\|\s*", " | ", text)
        text = re.sub(r"\s*\|\s*", " | ", text)
        return text

    @staticmethod
    def _normalize_ocr_noise(text: str) -> str:
        replacements = {
            "0m": "0m",
            "l0": "10",
            "rn": "m",
            "Il": "I",
            "1n": "in",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def clean(self, document: CommonDocument) -> CommonDocument:
        cleaned_text = document.content
        cleaned_text = self._remove_empty_documents(cleaned_text)
        cleaned_text = self._normalize_whitespace(cleaned_text)
        cleaned_text = self._remove_page_numbers(cleaned_text)
        cleaned_text = self._remove_duplicate_lines(cleaned_text)
        cleaned_text = self._remove_boilerplate(cleaned_text)
        cleaned_text = self._fix_broken_sentences(cleaned_text)
        cleaned_text = self._normalize_tables(cleaned_text)
        cleaned_text = self._normalize_ocr_noise(cleaned_text)

        document.content = cleaned_text
        document.metadata["cleaned"] = True
        document.metadata["cleaning_applied"] = [
            "empty_document_check",
            "whitespace_normalization",
            "page_number_removal",
            "duplicate_removal",
            "boilerplate_removal",
            "sentence_fixing",
            "table_normalization",
            "ocr_noise_cleanup",
        ]
        return document

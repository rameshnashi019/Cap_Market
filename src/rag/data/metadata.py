from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MetadataEnricher:
    """Build consistent metadata for normalized documents."""

    DOCUMENT_TYPES = {
        ".csv": "spreadsheet",
        ".docx": "word_document",
        ".html": "web_page",
        ".htm": "web_page",
        ".json": "structured_data",
        ".md": "markdown",
        ".pdf": "pdf",
        ".rtf": "rich_text",
        ".txt": "text",
        ".xml": "structured_data",
        ".yaml": "structured_data",
        ".yml": "structured_data",
    }

    @staticmethod
    def _format_date(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    @staticmethod
    def _title_from_content(file_path: Path, content: str) -> str:
        for line in content.splitlines():
            candidate = line.strip().lstrip("#").strip()
            if candidate and len(candidate) <= 200:
                return candidate
        return file_path.stem.replace("_", " ").replace("-", " ").strip()

    def enrich(
        self,
        file_path: str | Path,
        content: str,
        page_count: int | None = None,
        embedded_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path).resolve()
        stat = path.stat()
        extension = path.suffix.lower()
        embedded_metadata = embedded_metadata or {}

        title = str(embedded_metadata.get("title") or "").strip()
        if not title:
            title = self._title_from_content(path, content)

        modified_date = self._format_date(stat.st_mtime)
        created_date = self._format_date(stat.st_ctime)
        metadata: dict[str, Any] = {
            "source": str(path),
            "source_name": path.name,
            "title": title,
            "document_type": self.DOCUMENT_TYPES.get(extension, extension.lstrip(".")),
            "date": modified_date,
            "created_date": created_date,
            "modified_date": modified_date,
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime_ns,
            "extension": extension,
            "page_count": page_count,
        }

        for key in ("author", "subject"):
            value = embedded_metadata.get(key)
            if value:
                metadata[key] = str(value).strip()

        return metadata

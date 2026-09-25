from __future__ import annotations

import csv
import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from rag.config.settings import settings
from rag.data.loader import list_documents
from rag.data.metadata import MetadataEnricher

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency fallback
    PdfReader = None

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    fitz = None


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._chunks.append(text)

    def get_text(self) -> str:
        return "\n".join(self._chunks)


@dataclass
class CommonDocument:
    file_path: str
    file_name: str
    extension: str
    content: str
    status: str = "success"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentIngestionManager:
    """Manage document discovery, ingestion, and source-change tracking."""

    DEFAULT_EXTENSIONS = {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".html",
        ".htm",
        ".xml",
        ".yaml",
        ".yml",
        ".pdf",
        ".docx",
        ".rtf",
    }

    def __init__(
        self,
        source_dir: str | Path | None = None,
        supported_extensions: set[str] | None = None,
    ) -> None:
        self.source_dir = Path(source_dir).resolve() if source_dir else settings.docs_dir
        self.source_dir.mkdir(parents=True, exist_ok=True)

        self.supported_extensions = {
            ext.lower() for ext in (supported_extensions or self.DEFAULT_EXTENSIONS)
        }

        self.logger = logging.getLogger("rag.data.ingestion")
        self.metadata_enricher = MetadataEnricher()
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            log_dir = settings.log_dir
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_dir / "ingestion.log", encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(file_handler)

        self._known_documents: dict[str, int] = self._scan_documents()
        self.logger.info("Initialized document tracking for %s", self.source_dir)

    def list_documents(self) -> list[Path]:
        return list_documents(self.source_dir, self.supported_extensions)

    def _scan_documents(self) -> dict[str, int]:
        documents: dict[str, int] = {}
        for file_path in self.list_documents():
            try:
                documents[str(file_path.resolve())] = file_path.stat().st_mtime_ns
            except OSError as exc:
                self.logger.warning("Could not read metadata for %s: %s", file_path, exc)
        return documents

    def detect_changes(self) -> dict[str, list[Path]]:
        current_documents = self._scan_documents()

        added = [
            Path(path)
            for path in current_documents
            if path not in self._known_documents
        ]
        removed = [
            Path(path)
            for path in self._known_documents
            if path not in current_documents
        ]
        updated = [
            Path(path)
            for path, mtime in current_documents.items()
            if path in self._known_documents and self._known_documents[path] != mtime
        ]

        for file_path in added:
            self.logger.info("Document added: %s", file_path)
        for file_path in removed:
            self.logger.warning("Document removed: %s", file_path)
        for file_path in updated:
            self.logger.info("Document updated: %s", file_path)

        return {
            "added": sorted(added),
            "removed": sorted(removed),
            "updated": sorted(updated),
        }

    def sync_documents(self) -> dict[str, list[Path]]:
        changes = self.detect_changes()
        self._known_documents = self._scan_documents()

        if not any(changes.values()):
            self.logger.info("No document changes detected in %s", self.source_dir)
        return changes

    def _read_text_file(self, file_path: Path) -> str:
        encoding_candidates = ["utf-8", "utf-8-sig", "latin-1"]
        last_error: Exception | None = None

        for encoding in encoding_candidates:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _strip_html(self, content: str) -> str:
        parser = _HTMLTextExtractor()
        parser.feed(content)
        return parser.get_text()

    def _extract_pdf_text(self, file_path: Path) -> str:
        if PdfReader is not None:
            reader = PdfReader(str(file_path))
            pages: list[str] = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text:
                    pages.append(text)
            return "\n".join(pages)

        if fitz is not None:
            doc = fitz.open(str(file_path))
            pages = [page.get_text() for page in doc]
            return "\n".join(page for page in pages if page)

        raise RuntimeError("No PDF extraction backend available")

    def _pdf_metadata(self, file_path: Path) -> tuple[int | None, dict[str, Any]]:
        if PdfReader is None:
            return None, {}

        reader = PdfReader(str(file_path))
        raw_metadata = reader.metadata or {}
        embedded_metadata = {
            "title": raw_metadata.get("/Title", ""),
            "author": raw_metadata.get("/Author", ""),
            "subject": raw_metadata.get("/Subject", ""),
        }
        return len(reader.pages), embedded_metadata

    def _extract_docx_text(self, file_path: Path) -> str:
        try:
            with zipfile.ZipFile(file_path) as archive:
                xml_data = archive.read("word/document.xml")
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid DOCX file: {file_path}") from exc

        text_matches = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml_data.decode("utf-8", errors="ignore"))
        return "\n".join(re.sub(r"<.*?>", "", match) for match in text_matches)

    def extract_text(self, file_path: str | Path) -> str:
        path = Path(file_path).resolve()
        suffix = path.suffix.lower()

        try:
            if suffix in {".txt", ".md", ".rtf", ".yaml", ".yml", ".xml"}:
                return self._read_text_file(path)

            if suffix == ".csv":
                with path.open("r", newline="", encoding="utf-8", errors="ignore") as csv_file:
                    rows = list(csv.reader(csv_file))
                return "\n".join(", ".join(row) for row in rows)

            if suffix in {".json"}:
                with path.open("r", encoding="utf-8", errors="ignore") as json_file:
                    data = json.load(json_file)
                return json.dumps(data, ensure_ascii=False, indent=2)

            if suffix in {".html", ".htm"}:
                return self._strip_html(self._read_text_file(path))

            if suffix == ".pdf":
                return self._extract_pdf_text(path)

            if suffix == ".docx":
                return self._extract_docx_text(path)

            raise ValueError(f"Unsupported document type: {suffix}")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to ingest document: %s", path)
            raise exc

    def parse_document(self, file_path: str | Path) -> CommonDocument:
        path = Path(file_path).resolve()
        metadata: dict[str, Any] = {}

        try:
            content = self.extract_text(path)
            page_count: int | None = None
            embedded_metadata: dict[str, Any] = {}
            if path.suffix.lower() == ".pdf":
                page_count, embedded_metadata = self._pdf_metadata(path)
            metadata = self.metadata_enricher.enrich(
                path,
                content,
                page_count=page_count,
                embedded_metadata=embedded_metadata,
            )
            return CommonDocument(
                file_path=str(path),
                file_name=path.name,
                extension=path.suffix.lower(),
                content=content,
                metadata=metadata,
                status="success",
            )
        except Exception as exc:
            self.logger.exception("Failed to parse document into common format: %s", path)
            if not metadata:
                try:
                    metadata = self.metadata_enricher.enrich(path, "")
                except OSError:
                    metadata = {"source": str(path), "extension": path.suffix.lower()}
            return CommonDocument(
                file_path=str(path),
                file_name=path.name,
                extension=path.suffix.lower(),
                content="",
                status="failed",
                error=str(exc),
                metadata=metadata,
            )

    def ingest_all_documents(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for file_path in self.list_documents():
            parsed = self.parse_document(file_path)
            documents.append(
                {
                    "path": parsed.file_path,
                    "name": parsed.file_name,
                    "type": parsed.extension.lstrip("."),
                    "content": parsed.content,
                    "status": parsed.status,
                    **({"error": parsed.error} if parsed.error else {}),
                    "metadata": parsed.metadata,
                }
            )
        return documents

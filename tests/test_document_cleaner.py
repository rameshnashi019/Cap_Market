from pathlib import Path

from rag.data.cleaner import DocumentCleaner
from rag.data.ingestion import DocumentIngestionManager


def test_cleaner_removes_empty_and_duplicate_content(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    file_path = source_dir / "notes.txt"
    file_path.write_text("Hello world\nHello world\n\n   \n", encoding="utf-8")

    manager = DocumentIngestionManager(source_dir=source_dir)
    document = manager.parse_document(file_path)
    cleaned = DocumentCleaner().clean(document)

    assert cleaned.content == "Hello world"
    assert cleaned.metadata["cleaned"] is True


def test_cleaner_preserves_content_and_marks_cleaned(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    file_path = source_dir / "report.txt"
    file_path.write_text("   Sample   text  with   weird spacing\nPage 1\nSample text with weird spacing\n", encoding="utf-8")

    manager = DocumentIngestionManager(source_dir=source_dir)
    document = manager.parse_document(file_path)
    cleaned = DocumentCleaner().clean(document)

    assert "Sample text with weird spacing" in cleaned.content
    assert "Page 1" not in cleaned.content
    assert cleaned.metadata["cleaned"] is True

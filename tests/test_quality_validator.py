from pathlib import Path

from rag.data.cleaner import DocumentCleaner
from rag.data.ingestion import DocumentIngestionManager
from rag.data.validator import QualityValidator


def test_quality_validator_accepts_clean_document(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    file_path = source_dir / "report.txt"
    file_path.write_text("This is a valid document with clean text and useful content.", encoding="utf-8")

    manager = DocumentIngestionManager(source_dir=source_dir)
    document = manager.parse_document(file_path)
    cleaned = DocumentCleaner().clean(document)

    result = QualityValidator().validate(cleaned)

    assert result["is_valid"] is True
    assert result["checks"]["empty_text"] is True
    assert result["checks"]["duplicate_text"] is True


def test_quality_validator_rejects_empty_or_garbage_content(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    file_path = source_dir / "bad.txt"
    file_path.write_text("\x00\x01\x02\n\n", encoding="utf-8")

    manager = DocumentIngestionManager(source_dir=source_dir)
    document = manager.parse_document(file_path)
    cleaned = DocumentCleaner().clean(document)

    result = QualityValidator().validate(cleaned)

    assert result["is_valid"] is False
    assert "empty_text" in result["checks"]
    assert "garbage_characters" in result["checks"]

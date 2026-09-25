from pathlib import Path

from rag.data.ingestion import DocumentIngestionManager


def test_ingestion_enriches_document_metadata(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    file_path = source_dir / "project_notes.md"
    file_path.write_text("# Project Notes\n\nImportant project content.", encoding="utf-8")

    document = DocumentIngestionManager(source_dir=source_dir).parse_document(file_path)

    assert document.metadata["source"] == str(file_path.resolve())
    assert document.metadata["source_name"] == "project_notes.md"
    assert document.metadata["title"] == "Project Notes"
    assert document.metadata["document_type"] == "markdown"
    assert document.metadata["extension"] == ".md"
    assert document.metadata["page_count"] is None
    assert document.metadata["date"] == document.metadata["modified_date"]
    assert document.metadata["created_date"]
    assert document.metadata["modified_date"]


def test_metadata_title_falls_back_to_filename(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    file_path = source_dir / "annual-report.txt"
    file_path.write_text("\n\n", encoding="utf-8")

    document = DocumentIngestionManager(source_dir=source_dir).parse_document(file_path)

    assert document.metadata["title"] == "annual report"
    assert document.metadata["document_type"] == "text"

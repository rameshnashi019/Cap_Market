from pathlib import Path

from rag.data.ingestion import DocumentIngestionManager


def test_ingestion_detects_added_documents(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()

    manager = DocumentIngestionManager(source_dir=source_dir)

    file_path = source_dir / "sample.txt"
    file_path.write_text("hello world", encoding="utf-8")

    changes = manager.sync_documents()

    assert len(changes["added"]) == 1
    assert changes["added"][0].name == "sample.txt"


def test_ingestion_detects_removed_documents(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    file_path = source_dir / "old.txt"
    file_path.write_text("old content", encoding="utf-8")

    manager = DocumentIngestionManager(source_dir=source_dir)
    manager.sync_documents()

    file_path.unlink()
    changes = manager.sync_documents()

    assert len(changes["removed"]) == 1
    assert changes["removed"][0].name == "old.txt"


def test_common_document_representation_is_created(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    file_path = source_dir / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")

    manager = DocumentIngestionManager(source_dir=source_dir)
    document = manager.parse_document(file_path)

    assert document.file_name == "notes.txt"
    assert document.extension == ".txt"
    assert document.content == "hello world"
    assert document.status == "success"
    assert document.metadata["size_bytes"] > 0

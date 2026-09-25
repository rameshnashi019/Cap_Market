"""Document discovery utilities."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("rag.data.loader")


def list_documents(
    directory: str | Path,
    extensions: set[str] | None = None,
) -> list[Path]:
    """Return supported documents under a directory, recursively."""
    path = Path(directory).resolve()
    if not path.exists():
        logger.warning("Document directory does not exist: %s", path)
        return []

    supported = {ext.lower() for ext in (extensions or set())}
    files = [
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and (not supported or file_path.suffix.lower() in supported)
    ]
    return sorted(files)

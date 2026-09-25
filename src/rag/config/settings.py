from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _resolve_path(raw_path: str | None, fallback: Path) -> Path:
    candidate = Path(raw_path) if raw_path else fallback
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


@dataclass
class Settings:
    project_root: Path = Path(__file__).resolve().parents[3]
    src_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    vectorstore_dir: Path = project_root / "data" / "vectorstore"
    docs_dir: Path = project_root / "data" / "docs"
    log_dir: Path = project_root / "logs"
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    app_secret_key: str = "development-only-change-me"
    app_username: str = "admin"
    app_password: str = "change-me"
    jwt_expire_minutes: int = 60

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        raw_vector_path = os.getenv("CHROMA_PERSIST_DIRECTORY") or os.getenv("VECTORSTORE_PATH")
        default_vector_dir = self.project_root / "data" / "vectorstore"
        self.vectorstore_dir = _resolve_path(raw_vector_path, default_vector_dir)
        self.docs_dir = self.data_dir / "docs"
        self.log_dir = self.project_root / "logs"

        self.openai_model = os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL") or self.openai_model
        self.embedding_model = (
            os.getenv("HF_EMBEDDING_MODEL")
            or os.getenv("EMBEDDING_MODEL")
            or self.embedding_model
        )
        self.app_secret_key = os.getenv("JWT_SECRET_KEY") or os.getenv("APP_SECRET_KEY") or self.app_secret_key
        self.app_username = os.getenv("AUTH_USERNAME") or os.getenv("APP_USERNAME") or self.app_username
        self.app_password = os.getenv("AUTH_PASSWORD") or os.getenv("APP_PASSWORD") or self.app_password
        self.jwt_expire_minutes = int(os.getenv("JWT_EXPIRE_MINUTES", str(self.jwt_expire_minutes)))


settings = Settings()

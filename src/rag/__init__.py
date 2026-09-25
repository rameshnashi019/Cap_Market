"""RAG package root."""

from .config.settings import settings

__all__ = ["settings"]


def main() -> None:
    print("Hello from rag!")

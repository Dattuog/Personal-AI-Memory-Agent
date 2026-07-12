from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    @abstractmethod
    def add(self, id: str, embedding: list[float], text: str, metadata: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def query(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        """Return dicts containing id, text, metadata, and distance or score."""
        ...

    @abstractmethod
    def delete(self, id: str) -> None:
        ...

    @abstractmethod
    def list_all(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Return all stored memories for scans that are not query-driven."""
        ...

    @abstractmethod
    def update_metadata(self, id: str, metadata: dict[str, Any]) -> None:
        """Merge or overwrite metadata fields for a stored memory."""
        ...

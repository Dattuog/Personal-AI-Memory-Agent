from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    @abstractmethod
    def synthesize(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        """
        context_chunks contains retrieved memory dicts with text and metadata.
        Returns the synthesized answer as plain text.
        """
        ...

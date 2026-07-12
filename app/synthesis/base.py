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

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
        """Raw chat completion, no RAG context injection."""
        ...

    @abstractmethod
    def complete_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Return either a text result or a single tool call request."""
        ...

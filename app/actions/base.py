from abc import ABC, abstractmethod
from typing import Any


class Action(ABC):
    name: str
    description: str
    parameters_schema: dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Run the action and return a result dictionary."""
        ...

    def to_tool_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

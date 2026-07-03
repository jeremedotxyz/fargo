"""Tool definition.

A Tool pairs a JSON schema (what the model sees) with a plain Python
callable (what actually runs). Nothing fancier than that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the tool's arguments
    function: Callable[..., Any]

    def to_openai_schema(self) -> dict[str, Any]:
        """Return this tool in OpenAI's `tools=[...]` format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs: Any) -> Any:
        return self.function(**kwargs)

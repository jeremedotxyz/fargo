"""Tool registry.

Holds the set of tools available to an agent, builds the OpenAI-format
schema list to send with each request, and dispatches tool_call
invocations to the right function — catching and reporting errors
rather than crashing the agent loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from harness.tools.base import Tool

logger = logging.getLogger("harness.tools.registry")


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def dispatch(self, tool_call: Any) -> str:
        """Run a single OpenAI tool_call and return a string result.

        Never raises: tool errors are caught and returned as a string
        so the model can see what went wrong and try again or explain
        the failure to the user.
        """
        name = tool_call.function.name
        raw_args = tool_call.function.arguments or "{}"

        tool = self.get(name)
        if tool is None:
            logger.warning("Model requested unknown tool: %s", name)
            return f"Error: no tool named '{name}' is registered."

        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            logger.warning("Bad tool arguments for %s: %s", name, exc)
            return f"Error: could not parse arguments for '{name}': {exc}"

        try:
            result = tool.execute(**args)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, fed back to the model
            logger.warning("Tool '%s' raised an exception: %s", name, exc)
            return f"Error running tool '{name}': {exc}"

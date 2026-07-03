"""The agent loop.

    user message -> model -> (tool calls? -> execute -> feed results back -> model) -> final answer

Kept as one readable method (`run`) rather than spread across a state
machine, so you can see and modify the whole loop at a glance.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from harness.config import Config
from harness.llm_client import LLMClient
from harness.memory import ConversationMemory
from harness.tools.registry import ToolRegistry

logger = logging.getLogger("harness.agent")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. Use tools when they "
    "would give you a more accurate or current answer than reasoning alone. "
    "When you're done, give a direct, concise final answer."
)


class Agent:
    def __init__(
        self,
        config: Config | None = None,
        tool_registry: ToolRegistry | None = None,
        memory: ConversationMemory | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config or Config.from_env()
        self.tools = tool_registry or ToolRegistry()
        self.memory = memory or ConversationMemory(persist_path=self.config.memory_path)
        self.system_prompt = system_prompt
        self.on_tool_call = on_tool_call  # optional callback, e.g. for CLI printing

        self.llm = LLMClient(
            api_key=self.config.api_key,
            model=self.config.model,
            temperature=self.config.temperature,
        )

        if not self.memory.get_all():
            self.memory.add({"role": "system", "content": self.system_prompt})

    def run(self, user_message: str) -> str:
        """Send a user message through the agent loop and return the final text answer."""
        self.memory.add({"role": "user", "content": user_message})

        tool_schemas = self.tools.schemas() or None

        for iteration in range(self.config.max_iterations):
            message = self.llm.complete(self.memory.get_all(), tools=tool_schemas)

            # No tool calls -> this is the final answer.
            if not getattr(message, "tool_calls", None):
                content = message.content or ""
                self.memory.add({"role": "assistant", "content": content})
                return content

            # Record the assistant's tool-call request, then execute each one.
            self.memory.add(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tool_call in message.tool_calls:
                if self.on_tool_call:
                    import json

                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                    except Exception:  # noqa: BLE001
                        args = {}
                    self.on_tool_call(tool_call.function.name, args)

                result = self.tools.dispatch(tool_call)
                self.memory.add(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

            logger.debug("Completed tool round %d/%d", iteration + 1, self.config.max_iterations)

        # Safety cap hit: ask the model to wrap up without further tool calls.
        logger.warning("Max iterations (%d) reached; forcing a final answer.", self.config.max_iterations)
        self.memory.add(
            {
                "role": "user",
                "content": (
                    "You've reached the tool-call limit for this turn. "
                    "Please give your best final answer now without calling any more tools."
                ),
            }
        )
        final_message = self.llm.complete(self.memory.get_all(), tools=None)
        content = final_message.content or "(no answer produced)"
        self.memory.add({"role": "assistant", "content": content})
        return content

    def reset(self) -> None:
        """Clear conversation history back to just the system prompt."""
        self.memory.clear()
        self.memory.add({"role": "system", "content": self.system_prompt})

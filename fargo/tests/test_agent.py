"""Unit tests. Mocks the LLM client so no real API key or network call is needed.

Run with: pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness.agent import Agent  # noqa: E402
from harness.config import Config  # noqa: E402
from harness.memory import ConversationMemory  # noqa: E402
from harness.tools.base import Tool  # noqa: E402
from harness.tools.registry import ToolRegistry  # noqa: E402


def make_config() -> Config:
    return Config(api_key="test-key", model="gpt-4.1", max_iterations=4, temperature=0.0)


def fake_message(content: str | None = None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def fake_tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_direct_answer_no_tools():
    """If the model returns no tool_calls, run() returns its content immediately."""
    agent = Agent(config=make_config(), memory=ConversationMemory())
    agent.llm = MagicMock()
    agent.llm.complete.return_value = fake_message(content="42")

    result = agent.run("what is the answer?")

    assert result == "42"
    assert agent.llm.complete.call_count == 1


def test_single_tool_call_round_trip():
    """Model requests a tool, harness executes it, feeds result back, model answers."""
    calc = Tool(
        name="add",
        description="add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        function=lambda a, b: a + b,
    )
    registry = ToolRegistry([calc])
    agent = Agent(config=make_config(), tool_registry=registry, memory=ConversationMemory())
    agent.llm = MagicMock()

    tool_call = fake_tool_call("call_1", "add", '{"a": 2, "b": 3}')
    agent.llm.complete.side_effect = [
        fake_message(content=None, tool_calls=[tool_call]),
        fake_message(content="2 + 3 = 5"),
    ]

    result = agent.run("what is 2 + 3?")

    assert result == "2 + 3 = 5"
    assert agent.llm.complete.call_count == 2

    tool_messages = [m for m in agent.memory.get_all() if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "5"


def test_unknown_tool_returns_error_without_crashing():
    registry = ToolRegistry()
    result = registry.dispatch(fake_tool_call("call_1", "nonexistent", "{}"))
    assert "no tool named" in result


def test_tool_exception_is_caught():
    def boom():
        raise RuntimeError("kaboom")

    registry = ToolRegistry([Tool(name="boom", description="", parameters={"type": "object", "properties": {}}, function=boom)])
    result = registry.dispatch(fake_tool_call("call_1", "boom", "{}"))
    assert "kaboom" in result


def test_max_iterations_forces_final_answer():
    """If the model keeps requesting tools forever, the loop caps out and forces an answer."""
    noop = Tool(
        name="noop",
        description="does nothing",
        parameters={"type": "object", "properties": {}},
        function=lambda: "ok",
    )
    registry = ToolRegistry([noop])
    config = make_config()
    agent = Agent(config=config, tool_registry=registry, memory=ConversationMemory())
    agent.llm = MagicMock()

    endless_tool_call = fake_tool_call("call_x", "noop", "{}")
    # Always return a tool call, forcing the loop to hit max_iterations,
    # then the forced final call returns plain content.
    responses = [fake_message(content=None, tool_calls=[endless_tool_call])] * config.max_iterations
    responses.append(fake_message(content="giving up gracefully"))
    agent.llm.complete.side_effect = responses

    result = agent.run("loop forever")

    assert result == "giving up gracefully"
    assert agent.llm.complete.call_count == config.max_iterations + 1


def test_memory_persists_across_agent_instances(tmp_path):
    persist_path = tmp_path / "mem.json"
    mem1 = ConversationMemory(persist_path=str(persist_path))
    mem1.add({"role": "user", "content": "hello"})

    mem2 = ConversationMemory(persist_path=str(persist_path))
    assert mem2.get_all() == [{"role": "user", "content": "hello"}]

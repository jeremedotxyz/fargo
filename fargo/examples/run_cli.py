"""Interactive REPL for the harness.

Run with:
    python examples/run_cli.py

Type 'exit' or Ctrl+C to quit, 'reset' to clear conversation history.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running directly from the repo without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness.agent import Agent  # noqa: E402
from harness.tools.builtin import calculator_tool, clock_tool, make_file_reader_tool  # noqa: E402
from harness.tools.registry import ToolRegistry  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def print_tool_call(name: str, args: dict) -> None:
    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    print(f"  [tool call] {name}({arg_str})")


def main() -> None:
    registry = ToolRegistry(
        [calculator_tool, clock_tool, make_file_reader_tool(base_dir=".")]
    )
    agent = Agent(tool_registry=registry, on_tool_call=print_tool_call)

    print("llm-tool-harness — type 'exit' to quit, 'reset' to clear history.\n")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("(conversation cleared)")
            continue

        try:
            answer = agent.run(user_input)
            print(answer)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()

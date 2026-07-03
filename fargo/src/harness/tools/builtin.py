"""A handful of example tools so the harness works out of the box.

Replace these with real tools (web search, file I/O, your own APIs).
They're intentionally simple.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone
from pathlib import Path

from harness.tools.base import Tool

# --- calculator -------------------------------------------------------

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression (+ - * / % **)."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Could not evaluate '{expression}': {exc}"


calculator_tool = Tool(
    name="calculator",
    description="Evaluate a basic arithmetic expression. Supports + - * / % ** and parentheses.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The arithmetic expression to evaluate, e.g. '2847 * 193'",
            }
        },
        "required": ["expression"],
    },
    function=calculator,
)

# --- clock --------------------------------------------------------------


def get_current_time() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


clock_tool = Tool(
    name="get_current_time",
    description="Get the current date and time in UTC, ISO 8601 format.",
    parameters={"type": "object", "properties": {}, "required": []},
    function=get_current_time,
)

# --- file reader (sandboxed to a base directory) ------------------------


def make_file_reader_tool(base_dir: str = ".") -> Tool:
    """Build a file-reading tool sandboxed to `base_dir`.

    Returns a Tool rather than a bare function because the allowed
    directory is a closure parameter — instantiate per-agent as needed.
    """
    root = Path(base_dir).resolve()

    def read_file(path: str) -> str:
        target = (root / path).resolve()
        if root not in target.parents and target != root:
            return f"Error: '{path}' is outside the allowed directory."
        if not target.exists():
            return f"Error: '{path}' does not exist."
        if not target.is_file():
            return f"Error: '{path}' is not a file."
        try:
            return target.read_text(errors="replace")[:20_000]
        except OSError as exc:
            return f"Error reading '{path}': {exc}"

    return Tool(
        name="read_file",
        description=f"Read a text file's contents, relative to the sandboxed directory '{root}'.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file."}
            },
            "required": ["path"],
        },
        function=read_file,
    )

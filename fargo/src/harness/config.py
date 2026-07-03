"""Environment-based configuration for the harness.

Everything the agent needs to know at startup lives here. Reads from
process environment variables (populated from `.env` via python-dotenv
if present).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str = "gpt-4.1"
    max_iterations: int = 8
    temperature: float = 0.7
    memory_path: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return cls(
            api_key=api_key,
            model=os.getenv("HARNESS_MODEL", "gpt-4.1"),
            max_iterations=int(os.getenv("HARNESS_MAX_ITERATIONS", "8")),
            temperature=float(os.getenv("HARNESS_TEMPERATURE", "0.7")),
            memory_path=os.getenv("HARNESS_MEMORY_PATH") or None,
        )

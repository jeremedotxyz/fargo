"""Conversation memory.

Deliberately dumb: a list of OpenAI-format message dicts, optionally
persisted to a JSON file between runs. Replace with a vector store,
summarizer, or sliding window once your context needs outgrow this.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("harness.memory")


class ConversationMemory:
    def __init__(self, persist_path: str | None = None) -> None:
        self.persist_path = Path(persist_path) if persist_path else None
        self.messages: list[dict[str, Any]] = []
        if self.persist_path and self.persist_path.exists():
            self._load()

    def add(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        if self.persist_path:
            self._save()

    def extend(self, messages: list[dict[str, Any]]) -> None:
        self.messages.extend(messages)
        if self.persist_path:
            self._save()

    def get_all(self) -> list[dict[str, Any]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages = []
        if self.persist_path:
            self._save()

    def _save(self) -> None:
        assert self.persist_path is not None
        try:
            self.persist_path.write_text(json.dumps(self.messages, indent=2))
        except OSError as exc:
            logger.warning("Failed to persist memory to %s: %s", self.persist_path, exc)

    def _load(self) -> None:
        assert self.persist_path is not None
        try:
            self.messages = json.loads(self.persist_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load memory from %s: %s", self.persist_path, exc)
            self.messages = []

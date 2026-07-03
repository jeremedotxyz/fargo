"""Thin wrapper around the OpenAI chat completions API.

Kept deliberately small: one method in, one response out. Retries on
transient errors with exponential backoff. Swap this module out if you
want to point the harness at a different provider — the Agent only
depends on `LLMClient.complete()`'s return shape, not on OpenAI internals.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

logger = logging.getLogger("harness.llm_client")


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_base_delay: float = 1.5,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Call the model once and return the raw completion message.

        Retries on rate limits / connection errors / 5xx with exponential
        backoff. Raises on the final failed attempt.
        """
        attempt = 0
        while True:
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message

            except (RateLimitError, APIConnectionError) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    logger.error("Max retries exceeded: %s", exc)
                    raise
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Transient error (%s), retrying in %.1fs [attempt %d/%d]",
                    exc,
                    delay,
                    attempt,
                    self.max_retries,
                )
                time.sleep(delay)

            except APIError as exc:
                # 5xx errors are worth retrying; 4xx (bad request, auth) are not.
                status = getattr(exc, "status_code", None)
                attempt += 1
                if status and 500 <= status < 600 and attempt <= self.max_retries:
                    delay = self.retry_base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Server error %s, retrying in %.1fs [attempt %d/%d]",
                        status,
                        delay,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(delay)
                    continue
                logger.error("Non-retryable API error: %s", exc)
                raise

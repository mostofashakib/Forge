"""Deterministic LLM clients for tests. Keyed by schema class name."""
from __future__ import annotations

from pydantic import BaseModel


class MockLLMClient:
    """Deterministic mock for testing. Keyed by schema class name."""

    def __init__(self, responses: dict[str, BaseModel]) -> None:
        self._responses = responses

    def extract(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        key = schema.__name__
        if key not in self._responses:
            raise ValueError(f"No mock response for schema '{key}'")
        return self._responses[key]


class RetryMockLLMClient:
    """Fails ``fail_times`` times then succeeds. For testing retry logic."""

    def __init__(self, fail_times: int, then_return: dict[str, BaseModel]) -> None:
        self._fail_times = fail_times
        self._then_return = then_return
        self.call_count = 0

    def extract(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        while True:
            self.call_count += 1
            if self.call_count <= self._fail_times:
                continue
            key = schema.__name__
            if key not in self._then_return:
                raise ValueError(f"No mock response for '{key}'")
            return self._then_return[key]

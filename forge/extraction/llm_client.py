from __future__ import annotations
from typing import Protocol, runtime_checkable
from pydantic import BaseModel


@runtime_checkable
class LLMClient(Protocol):
    def extract(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel: ...


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-4-6", max_retries: int = 3, max_tokens: int = 8192) -> None:
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_retries = max_retries
        self._max_tokens = max_tokens

    def extract(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        import anthropic
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                extra = f"\n\nPrevious attempt failed: {last_error}" if last_error else ""
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system + extra,
                    messages=[{"role": "user", "content": user}],
                    tools=[{
                        "name": "extract",
                        "description": (
                            f"Return your {schema.__name__} extraction results. "
                            "Populate ALL required fields based on the description."
                        ),
                        "input_schema": schema.model_json_schema(),
                    }],
                    tool_choice={"type": "tool", "name": "extract"},
                )
                tool_block = next(b for b in response.content if b.type == "tool_use")
                return schema.model_validate(tool_block.input)
            except Exception as e:
                last_error = e
                if attempt == self._max_retries - 1:
                    raise RuntimeError(
                        f"LLM extraction failed after {self._max_retries} attempts: {e}"
                    ) from e
        raise RuntimeError("unreachable")


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
    """Fails `fail_times` times then succeeds. For testing retry logic."""

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

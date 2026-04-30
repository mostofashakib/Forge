from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
from pydantic import BaseModel


@runtime_checkable
class LLMClient(Protocol):
    def extract(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel: ...


def _inline_refs(obj: Any, defs: dict[str, Any]) -> Any:
    """Recursively resolve $ref pointers so the schema has no $defs."""
    if isinstance(obj, list):
        return [_inline_refs(item, defs) for item in obj]
    if isinstance(obj, dict):
        if "$ref" in obj:
            ref_name = obj["$ref"].split("/")[-1]
            return _inline_refs(defs[ref_name], defs)
        return {k: _inline_refs(v, defs) for k, v in obj.items() if k != "title"}
    return obj


def _flat_schema(schema_cls: type[BaseModel]) -> dict[str, Any]:
    """Return a flattened JSON schema with $defs inlined and title keys stripped."""
    raw = schema_cls.model_json_schema()
    defs = raw.pop("$defs", {})
    flat = _inline_refs(raw, defs)
    flat.pop("title", None)
    return flat


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-4-6", max_retries: int = 3, max_tokens: int = 8192) -> None:
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_retries = max_retries
        self._max_tokens = max_tokens

    def extract(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        input_schema = _flat_schema(schema)
        last_error: Exception | None = None
        budget = self._max_tokens
        attempts_remaining = self._max_retries

        while attempts_remaining > 0:
            extra_headers = (
                {"anthropic-beta": "output-128k-2025-02-19"}
                if budget > 8192
                else {}
            )
            try:
                extra = f"\n\nPrevious attempt failed: {last_error}" if last_error else ""
                # Always stream: the SDK requires it for requests that may exceed
                # 10 minutes, and get_final_message() returns the same Message object.
                with self._client.messages.stream(
                    model=self._model,
                    max_tokens=budget,
                    system=system + extra,
                    messages=[{"role": "user", "content": user}],
                    tools=[{
                        "name": "extract",
                        "description": (
                            f"Return your {schema.__name__} extraction results. "
                            "Populate ALL required fields based on the description."
                        ),
                        "input_schema": input_schema,
                    }],
                    tool_choice={"type": "tool", "name": "extract"},
                    extra_headers=extra_headers,
                ) as stream:
                    response = stream.get_final_message()

                # Response truncated — double the budget and retry without burning an attempt.
                if response.stop_reason == "max_tokens":
                    if budget >= 128_000:
                        raise RuntimeError(
                            f"Response still truncated at maximum budget (128k tokens); "
                            f"schema={schema.__name__}"
                        )
                    budget = min(budget * 2, 128_000)
                    last_error = ValueError(
                        f"stop_reason=max_tokens; budget doubled to {budget}"
                    )
                    continue  # retry without decrementing attempts_remaining

                tool_block = next(b for b in response.content if b.type == "tool_use")
                if not tool_block.input:
                    raise ValueError(
                        f"Model returned empty tool input "
                        f"(stop_reason={response.stop_reason}, model={self._model})"
                    )
                return schema.model_validate(tool_block.input)
            except Exception as e:
                last_error = e
                attempts_remaining -= 1
                if attempts_remaining == 0:
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

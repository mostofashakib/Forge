from __future__ import annotations
import json
import os
from typing import Any, Protocol, runtime_checkable
from pydantic import BaseModel


@runtime_checkable
class LLMClient(Protocol):
    def extract(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel: ...


class LLMPromptFormatter:
    """Adds one consistent, explicit structured-output contract to every LLM prompt."""

    @staticmethod
    def structured(system: str, schema: type[BaseModel]) -> str:
        if not system.strip():
            raise ValueError("LLM system instruction cannot be empty")
        fields = ", ".join(schema.model_fields) or "no fields"
        return (
            f"{system.rstrip()}\n\n"
            "OUTPUT FORMAT (required): Return exactly one structured result through "
            "the configured extraction mechanism; do not return markdown or free-form text. "
            f"The result must validate as {schema.__name__} with these top-level fields: "
            f"{fields}. Populate every required field and use the schema's declared types."
        )


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicClient:
    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_retries: int = 3,
        max_tokens: int = 8192,
    ) -> None:
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_retries = max_retries
        self._max_tokens = max_tokens

    def _stream_extract(self, system: str, messages: list, schema: type[BaseModel]) -> BaseModel:
        system = LLMPromptFormatter.structured(system, schema)
        input_schema = _flat_schema(schema)
        last_error: Exception | None = None
        budget = self._max_tokens
        attempts_remaining = self._max_retries

        while attempts_remaining > 0:
            extra_headers = (
                {"anthropic-beta": "output-128k-2025-02-19"} if budget > 8192 else {}
            )
            try:
                extra = f"\n\nPrevious attempt failed: {last_error}" if last_error else ""
                with self._client.messages.stream(
                    model=self._model,
                    max_tokens=budget,
                    system=system + extra,
                    messages=messages,
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

                if response.stop_reason == "max_tokens":
                    if budget >= 128_000:
                        raise RuntimeError(
                            f"Response still truncated at maximum budget (128k tokens); "
                            f"schema={schema.__name__}"
                        )
                    budget = min(budget * 2, 128_000)
                    last_error = ValueError(f"stop_reason=max_tokens; budget doubled to {budget}")
                    continue

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

    def extract(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        return self._stream_extract(
            system=system,
            messages=[{"role": "user", "content": user}],
            schema=schema,
        )

    def extract_with_image(
        self,
        system: str,
        user: str,
        image_b64: str,
        schema: type[BaseModel],
        media_type: str = "image/png",
    ) -> BaseModel:
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                },
                {"type": "text", "text": user},
            ],
        }]
        return self._stream_extract(system=system, messages=messages, schema=schema)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaClient:
    """LLM client backed by a local Ollama instance.

    Uses Ollama's structured-output mode (``format=<json_schema>``) so the
    model is constrained to emit valid JSON matching the Pydantic schema.
    Works with any model served by Ollama; defaults to ``gemma4:26b``.

    Environment variables (all optional):
        OLLAMA_BASE_URL   Base URL of the Ollama server (default: http://localhost:11434)
    """

    def __init__(
        self,
        model: str = "gemma4:26b",
        max_retries: int = 3,
        max_tokens: int = 8192,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._max_tokens = max_tokens
        self._base_url = base_url

    def extract(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        import ollama  # lazy import — only required when Ollama is the active provider

        json_schema = _flat_schema(schema)
        system = LLMPromptFormatter.structured(system, schema)
        last_error: Exception | None = None
        client = ollama.Client(host=self._base_url)

        for _ in range(self._max_retries):
            try:
                extra = f"\n\nPrevious attempt failed: {last_error}" if last_error else ""
                response = client.chat(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system + extra},
                        {"role": "user", "content": user},
                    ],
                    format=json_schema,
                    options={"num_predict": self._max_tokens},
                )
                data = json.loads(response.message.content)
                return schema.model_validate(data)
            except Exception as e:
                last_error = e

        raise RuntimeError(
            f"Ollama extraction failed after {self._max_retries} attempts "
            f"(model={self._model}): {last_error}"
        ) from last_error


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ANTHROPIC_DEFAULT  = "claude-haiku-4-5-20251001"
_ANTHROPIC_CAPABLE  = "claude-sonnet-4-6"
_OLLAMA_DEFAULT     = "gemma4:26b"


def get_client(
    max_tokens: int = 8192,
    max_retries: int = 3,
    *,
    capable: bool = False,
    model: str | None = None,
    provider: str | None = None,
) -> LLMClient:
    """Return an LLM client configured from environment variables.

    Provider selection (in priority order):
        1. ``provider`` argument
        2. ``FORGE_LLM_PROVIDER`` env var  (default: ``anthropic``)

    Model selection (in priority order):
        1. ``model`` argument
        2. ``FORGE_LLM_MODEL_CAPABLE`` / ``FORGE_LLM_MODEL`` env vars
        3. Built-in defaults per provider

    Args:
        max_tokens:  Token budget passed to the underlying client.
        max_retries: Number of retry attempts on failure.
        capable:     When True, prefer the more powerful model tier
                     (e.g. Sonnet over Haiku for Anthropic). Ignored for
                     Ollama because a single model serves all tiers.
        model:       Explicit model override; skips env-var lookup.
        provider:    Explicit provider override; skips env-var lookup.

    Supported providers:
        anthropic — Anthropic API (requires ``ANTHROPIC_API_KEY``)
        ollama    — Local Ollama server (requires ``ollama`` package)
    """
    p = (provider or os.environ.get("FORGE_LLM_PROVIDER", "anthropic")).lower()

    if model is None:
        if capable:
            model = os.environ.get(
                "FORGE_LLM_MODEL_CAPABLE",
                _ANTHROPIC_CAPABLE if p == "anthropic" else _OLLAMA_DEFAULT,
            )
        else:
            model = os.environ.get(
                "FORGE_LLM_MODEL",
                _ANTHROPIC_DEFAULT if p == "anthropic" else _OLLAMA_DEFAULT,
            )

    if p == "anthropic":
        return AnthropicClient(model=model, max_tokens=max_tokens, max_retries=max_retries)

    if p == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaClient(
            model=model, max_tokens=max_tokens, max_retries=max_retries, base_url=base_url
        )

    raise ValueError(f"Unknown LLM provider: {p!r}. Valid: anthropic, ollama")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

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

"""Unit tests for the Ollama request construction in fleet.agents.model_adapters."""

import json

from fleet.agents.model_adapters import OllamaAdapter


def build_payload(format_schema=None) -> dict:
    adapter = OllamaAdapter("http://127.0.0.1:11434")
    request = adapter._build_request("prompt", "gemma4:26b", format_schema)
    return json.loads(request.data.decode("utf-8"))


def test_thinking_is_disabled():
    # Thinking-capable models (gemma4) default to think=true on /api/generate,
    # and Ollama's structured-output grammar corrupts the answer
    payload = build_payload()
    assert payload["think"] is False


def test_format_schema_is_forwarded():
    schema = {"type": "object"}
    assert build_payload(schema)["format"] == schema
    assert build_payload(None)["format"] == "json"

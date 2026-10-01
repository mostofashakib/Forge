"""Listing the open-source models the local Ollama server has pulled."""
from __future__ import annotations

import httpx
import pytest

from backend.app.services.ollama_catalog import OllamaUnavailable, list_ollama_models

TAGS = {
    "models": [
        {"name": "qwen3:32b", "size": 20_000_000_000, "details": {"parameter_size": "32.8B"}},
        {"name": "glm-4.7-flash:latest", "size": 19_000_000_000, "details": {"parameter_size": "29.9B"}},
        {"name": "glm-5.2:cloud", "size": 0, "details": {"parameter_size": "756b"}},
    ]
}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_models_are_listed_by_name_with_their_family():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=TAGS)

    models = list_ollama_models("http://ollama:11434", client=_client(handler))

    assert seen == ["http://ollama:11434/api/tags"]
    assert models == [
        {"name": "glm-4.7-flash:latest", "family": "glm", "parameters": "29.9B", "cloud": False},
        {"name": "glm-5.2:cloud", "family": "glm", "parameters": "756b", "cloud": True},
        {"name": "qwen3:32b", "family": "qwen", "parameters": "32.8B", "cloud": False},
    ]


def test_a_server_with_no_models_returns_an_empty_list():
    models = list_ollama_models("http://ollama:11434", client=_client(lambda r: httpx.Response(200, json={"models": []})))

    assert models == []


def test_an_unreachable_server_raises_with_the_url():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(OllamaUnavailable, match="ollama:11434"):
        list_ollama_models("http://ollama:11434", client=_client(handler))


def test_a_server_error_raises():
    with pytest.raises(OllamaUnavailable):
        list_ollama_models("http://ollama:11434", client=_client(lambda r: httpx.Response(500)))

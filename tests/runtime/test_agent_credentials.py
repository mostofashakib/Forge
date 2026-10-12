"""LLM agents get their keys from the credential registry."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from forge.credentials import CredentialNotConfigured
from forge.runtime.agents.factory import make_agent


def test_the_anthropic_agent_passes_the_configured_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-configured")
    with patch("forge.runtime.agents.anthropic_agent.anthropic") as sdk:
        make_agent("anthropic:claude-sonnet-4-6")

    sdk.Anthropic.assert_called_once_with(api_key="sk-ant-configured")


def test_the_anthropic_agent_refuses_the_placeholder(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock_key")
    with patch("forge.runtime.agents.anthropic_agent.anthropic"):
        with pytest.raises(CredentialNotConfigured, match="ANTHROPIC_API_KEY"):
            make_agent("anthropic:claude-sonnet-4-6")


def test_the_openai_agent_passes_the_configured_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-configured")
    with patch("forge.runtime.agents.openai_agent.openai") as sdk:
        make_agent("openai:gpt-4o")

    sdk.OpenAI.assert_called_once_with(api_key="sk-openai-configured")


def test_the_openai_agent_refuses_a_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("forge.runtime.agents.openai_agent.openai"):
        with pytest.raises(CredentialNotConfigured, match="OPENAI_API_KEY"):
            make_agent("openai:gpt-4o")


def test_the_vllm_agent_never_forwards_the_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-configured")
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box:8000/v1")
    with patch("forge.runtime.agents.openai_agent.openai") as sdk:
        sdk.OpenAI.return_value = MagicMock()
        make_agent("vllm:qwen")

    kwargs = sdk.OpenAI.call_args.kwargs
    assert kwargs["base_url"] == "http://gpu-box:8000/v1"
    assert kwargs["api_key"] != "sk-openai-configured"


def test_the_vllm_agent_runs_without_an_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("forge.runtime.agents.openai_agent.openai") as sdk:
        make_agent("vllm:qwen")

    assert sdk.OpenAI.call_args.kwargs["api_key"]

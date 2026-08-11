import pytest
from pydantic import BaseModel
from forge.extraction.llm_client import MockLLMClient, LLMClient
from forge.extraction.schemas import EntityDef, FieldDef
from forge.extraction.llm_client import LLMPromptFormatter


class _SimpleSchema(BaseModel):
    value: str


def test_mock_returns_predefined_response():
    response = _SimpleSchema(value="hello")
    client = MockLLMClient({"_SimpleSchema": response})
    result = client.extract("system", "user", _SimpleSchema)
    assert result.value == "hello"


def test_mock_raises_for_unknown_schema():
    client = MockLLMClient({})
    with pytest.raises(ValueError, match="No mock response"):
        client.extract("system", "user", _SimpleSchema)


def test_mock_client_satisfies_llmclient_protocol():
    client = MockLLMClient({"_SimpleSchema": _SimpleSchema(value="x")})
    assert isinstance(client, LLMClient)


def test_mock_retry_client_fails_then_succeeds():
    from forge.extraction.llm_client import RetryMockLLMClient
    response = _SimpleSchema(value="ok")
    client = RetryMockLLMClient(fail_times=2, then_return={"_SimpleSchema": response})
    result = client.extract("s", "u", _SimpleSchema)
    assert result.value == "ok"
    assert client.call_count == 3


def test_anthropic_client_default_max_tokens():
    from forge.extraction.llm_client import AnthropicClient
    client = AnthropicClient()
    assert client._max_tokens == 8192


def test_anthropic_client_custom_max_tokens():
    from forge.extraction.llm_client import AnthropicClient
    client = AnthropicClient(max_tokens=2048)
    assert client._max_tokens == 2048


def test_prompt_formatter_adds_explicit_output_contract():
    prompt = LLMPromptFormatter.structured("Perform the task carefully.", _SimpleSchema)
    assert "OUTPUT FORMAT (required)" in prompt
    assert "_SimpleSchema" in prompt
    assert "value" in prompt


def test_prompt_formatter_rejects_missing_instruction():
    with pytest.raises(ValueError, match="cannot be empty"):
        LLMPromptFormatter.structured("  ", _SimpleSchema)


# ---------------------------------------------------------------------------
# Generation vs. judge model resolution
# ---------------------------------------------------------------------------

def test_generation_models_reports_both_tiers(monkeypatch):
    from forge.extraction.llm_client import generation_models

    monkeypatch.setenv("FORGE_LLM_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-4-6")

    assert generation_models() == (
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
    )


def test_generation_models_does_not_duplicate_a_shared_tier(monkeypatch):
    from forge.extraction.llm_client import generation_models

    monkeypatch.setenv("FORGE_LLM_MODEL", "gemma4:26b")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "gemma4:26b")

    assert generation_models() == ("gemma4:26b",)


def test_judge_client_uses_the_judge_model_not_the_generation_model(monkeypatch):
    from forge.extraction.llm_client import get_judge_client

    monkeypatch.setenv("FORGE_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("FORGE_LLM_MODEL", "gemma4:26b")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "llama3.1:8b")

    client = get_judge_client()

    assert client._model == "llama3.1:8b"
    assert client._model != "gemma4:26b"


def test_judge_client_falls_back_to_the_generation_model_when_unset(monkeypatch):
    """No judge configured is not an error — it is a non-independent default."""
    from forge.extraction.llm_client import get_judge_client

    monkeypatch.setenv("FORGE_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("FORGE_LLM_MODEL", "gemma4:26b")
    monkeypatch.delenv("FORGE_JUDGE_MODEL", raising=False)

    assert get_judge_client()._model == "gemma4:26b"


def test_judge_client_can_use_a_different_provider(monkeypatch):
    from forge.extraction.llm_client import OllamaClient, get_judge_client

    monkeypatch.setenv("FORGE_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("FORGE_JUDGE_PROVIDER", "ollama")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "llama3.1:8b")

    assert isinstance(get_judge_client(), OllamaClient)


def test_judge_client_rejects_an_unknown_judge_provider(monkeypatch):
    from forge.extraction.llm_client import get_judge_client

    monkeypatch.setenv("FORGE_JUDGE_PROVIDER", "not-a-provider")

    with pytest.raises(ValueError):
        get_judge_client()


# ---------------------------------------------------------------------------
# Additional providers for quorum members
# ---------------------------------------------------------------------------

def test_openai_provider_builds_an_openai_client():
    from forge.extraction.llm_client import OpenAIClient, get_client

    client = get_client(provider="openai", model="gpt-4o")

    assert isinstance(client, OpenAIClient)
    assert client._model == "gpt-4o"


def test_gemini_provider_builds_a_gemini_client():
    from forge.extraction.llm_client import GeminiClient, get_client

    client = get_client(provider="gemini", model="gemini-2.0-flash")

    assert isinstance(client, GeminiClient)
    assert client._model == "gemini-2.0-flash"


def test_openai_and_gemini_clients_are_not_interchangeable():
    from forge.extraction.llm_client import GeminiClient, get_client

    assert not isinstance(get_client(provider="openai", model="gpt-4o"), GeminiClient)


def test_new_providers_default_their_model_when_none_is_given(monkeypatch):
    from forge.extraction.llm_client import get_client

    monkeypatch.delenv("FORGE_LLM_MODEL", raising=False)

    assert get_client(provider="openai")._model
    assert get_client(provider="gemini")._model


def test_unknown_provider_error_lists_every_supported_provider():
    from forge.extraction.llm_client import get_client

    with pytest.raises(ValueError) as excinfo:
        get_client(provider="not-a-provider")
    message = str(excinfo.value)
    for provider in ("anthropic", "ollama", "openai", "gemini"):
        assert provider in message


def test_building_a_provider_client_does_not_require_its_sdk():
    """Constructing a member must not import a package that may not be installed."""
    from forge.extraction.llm_client import get_client

    get_client(provider="openai", model="gpt-4o")
    get_client(provider="gemini", model="gemini-2.0-flash")


def test_generation_models_defaults_follow_the_configured_provider(monkeypatch):
    """A non-Anthropic generator must not report Ollama's default model."""
    from forge.extraction.llm_client import generation_models

    monkeypatch.setenv("FORGE_LLM_PROVIDER", "openai")
    monkeypatch.delenv("FORGE_LLM_MODEL", raising=False)
    monkeypatch.delenv("FORGE_LLM_MODEL_CAPABLE", raising=False)

    models = generation_models()

    assert models == ("gpt-4o",)
    assert "gemma4:26b" not in models

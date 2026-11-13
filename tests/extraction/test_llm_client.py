import pytest
from pydantic import BaseModel
from forge.extraction.llm_client import LLMClient
from tests.fake_llm import MockLLMClient
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
    from tests.fake_llm import RetryMockLLMClient
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


# ---------------------------------------------------------------------------
# Retries: only failures that can succeed on a second try are retried
# ---------------------------------------------------------------------------

class _StatusError(Exception):
    """Shaped like the provider SDKs' HTTP errors, which carry status_code."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _AlwaysFailingMessages:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def stream(self, **_kwargs):
        self.calls += 1
        raise self.error


def _anthropic_failing_with(error: Exception):
    from types import SimpleNamespace
    from forge.extraction.llm_client import AnthropicClient

    client = AnthropicClient(max_retries=3)
    messages = _AlwaysFailingMessages(error)
    client._client = SimpleNamespace(messages=messages)
    return client, messages


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(status):
    from pydantic import BaseModel

    class _Out(BaseModel):
        value: str

    client, messages = _anthropic_failing_with(_StatusError(status))
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        client.extract(system="s", user="u", schema=_Out)
    assert messages.calls == 1


@pytest.mark.parametrize("error", [_StatusError(429), _StatusError(529), ValueError("bad tool input")])
def test_transient_and_validation_failures_are_still_retried(error):
    from pydantic import BaseModel

    class _Out(BaseModel):
        value: str

    client, messages = _anthropic_failing_with(error)
    with pytest.raises(RuntimeError):
        client.extract(system="s", user="u", schema=_Out)
    assert messages.calls == 3


# ---------------------------------------------------------------------------
# Credentials come from the registry, never a placeholder
# ---------------------------------------------------------------------------

def test_anthropic_client_sends_the_configured_key(monkeypatch):
    from forge.extraction.llm_client import AnthropicClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-configured")

    assert AnthropicClient()._sdk().api_key == "sk-ant-configured"


def test_anthropic_client_refuses_the_placeholder_key_by_name(monkeypatch):
    from forge.credentials import CredentialNotConfigured
    from forge.extraction.llm_client import AnthropicClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock_key")
    client = AnthropicClient()

    with pytest.raises(CredentialNotConfigured, match="ANTHROPIC_API_KEY"):
        client.extract("system", "user", _SimpleSchema)


def test_openai_client_refuses_the_placeholder_key_before_any_request(monkeypatch):
    from forge.credentials import CredentialNotConfigured
    from forge.extraction.llm_client import OpenAIClient

    monkeypatch.setenv("OPENAI_API_KEY", "mock_key")

    with pytest.raises(CredentialNotConfigured, match="OPENAI_API_KEY"):
        OpenAIClient(model="gpt-4o", max_retries=1).extract("system", "user", _SimpleSchema)


def test_openai_client_passes_the_configured_key(monkeypatch):
    import openai
    from forge.extraction.llm_client import OpenAIClient

    seen = {}

    class _Refused(Exception):
        status_code = 401

    def _client(**kwargs):
        seen.update(kwargs)
        raise _Refused("stop before the network")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-configured")
    monkeypatch.setattr(openai, "OpenAI", _client)

    with pytest.raises(_Refused):
        OpenAIClient(model="gpt-4o", max_retries=1).extract("system", "user", _SimpleSchema)
    assert seen == {"api_key": "sk-openai-configured"}


def test_ollama_extraction_turns_thinking_off(monkeypatch):
    """A thinking model spends num_predict on hidden reasoning and returns truncated JSON."""
    import sys
    import types
    from forge.extraction.llm_client import OllamaClient

    calls = []

    class _Client:
        def __init__(self, host, **kwargs):
            pass

        def chat(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(message=types.SimpleNamespace(content='{"value": "ok"}'))

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(Client=_Client))

    result = OllamaClient(model="gemma4:26b").extract("system", "user", _SimpleSchema)

    assert result.value == "ok"
    assert calls[0]["think"] is False


def _ollama_recording_client(monkeypatch, *, fail_with: Exception | None = None) -> list[dict]:
    import sys
    import types

    built: list[dict] = []

    class _Client:
        def __init__(self, host, **kwargs):
            built.append(kwargs)

        def chat(self, **kwargs):
            if fail_with is not None:
                raise fail_with
            return types.SimpleNamespace(message=types.SimpleNamespace(content='{"value": "ok"}'))

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(Client=_Client))
    return built


def test_ollama_requests_carry_a_timeout_so_a_stalled_model_cannot_hang_the_caller(monkeypatch):
    from forge.extraction.llm_client import OLLAMA_REQUEST_TIMEOUT_S, OllamaClient

    built = _ollama_recording_client(monkeypatch)

    OllamaClient(model="gemma4:26b").extract("system", "user", _SimpleSchema)

    assert built == [{"timeout": OLLAMA_REQUEST_TIMEOUT_S}]
    assert 0 < OLLAMA_REQUEST_TIMEOUT_S < float("inf")


def test_an_ollama_request_that_keeps_timing_out_fails_after_the_retries(monkeypatch):
    import httpx

    from forge.extraction.llm_client import OllamaClient

    _ollama_recording_client(monkeypatch, fail_with=httpx.ReadTimeout("model stalled"))

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        OllamaClient(model="gemma4:26b", max_retries=2).extract("system", "user", _SimpleSchema)


def test_flat_schema_keeps_a_field_named_title():
    """Schema metadata titles go, but a real field called `title` stays in properties."""
    from forge.extraction.llm_client import _flat_schema

    class _Source(BaseModel):
        title: str
        url: str

    class _Wrapper(BaseModel):
        source: _Source

    schema = _flat_schema(_Wrapper)
    source = schema["properties"]["source"]

    assert set(source["required"]) <= set(source["properties"])
    assert source["properties"]["title"] == {"type": "string"}
    assert "title" not in schema


# ---------------------------------------------------------------------------
# Retries running out is its own error, so callers can treat one bad answer
# as a lost item while a permanent failure (bad key) still stops them.
# ---------------------------------------------------------------------------

def test_anthropic_retries_running_out_raises_llm_retries_exhausted():
    from pydantic import BaseModel

    from forge.extraction.llm_client import LLMRetriesExhausted

    class _Out(BaseModel):
        value: str

    client, _messages = _anthropic_failing_with(ValueError("bad tool input"))
    with pytest.raises(LLMRetriesExhausted, match="after 3 attempts"):
        client.extract(system="s", user="u", schema=_Out)


def test_a_permanent_failure_is_not_reported_as_retries_running_out():
    from pydantic import BaseModel

    from forge.extraction.llm_client import LLMRetriesExhausted

    class _Out(BaseModel):
        value: str

    client, _messages = _anthropic_failing_with(_StatusError(401))
    with pytest.raises(RuntimeError) as raised:
        client.extract(system="s", user="u", schema=_Out)
    assert not isinstance(raised.value, LLMRetriesExhausted)


def test_ollama_returning_broken_json_every_time_raises_llm_retries_exhausted(monkeypatch):
    import sys
    import types

    from forge.extraction.llm_client import LLMRetriesExhausted, OllamaClient

    class _Client:
        def __init__(self, host, **kwargs):
            pass

        def chat(self, **kwargs):
            return types.SimpleNamespace(message=types.SimpleNamespace(content='{"value" "ok"}'))

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(Client=_Client))

    with pytest.raises(LLMRetriesExhausted, match="Expecting ':' delimiter"):
        OllamaClient(model="qwen3.8:27b", max_retries=2).extract("system", "user", _SimpleSchema)

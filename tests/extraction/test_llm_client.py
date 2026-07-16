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

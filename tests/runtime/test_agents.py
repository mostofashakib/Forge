import pytest
import yaml
import tempfile
import os
from unittest.mock import MagicMock, patch
from forge.runtime.agents.base import AgentAdapter
from forge.runtime.agents.random_agent import RandomAgent
from forge.runtime.agents.scripted_agent import ScriptedAgent
from forge.runtime.agents.factory import make_agent
from forge.runtime.errors import AgentError


ACTION_TYPES = frozenset(["offer_refund", "close_ticket", "escalate"])


def test_random_agent_returns_valid_action():
    agent = RandomAgent()
    for _ in range(20):
        result = agent.act({}, ACTION_TYPES)
        assert result["type"] in ACTION_TYPES


def test_random_agent_single_action_type():
    agent = RandomAgent()
    result = agent.act({}, frozenset(["only_action"]))
    assert result["type"] == "only_action"


def test_scripted_agent_cycles_through_sequence():
    actions = [
        {"type": "offer_refund", "ticket_id": "t_1"},
        {"type": "close_ticket"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(actions, f)
        path = f.name
    try:
        agent = ScriptedAgent(path)
        r1 = agent.act({}, ACTION_TYPES)
        assert r1 == {"type": "offer_refund", "ticket_id": "t_1"}
        r2 = agent.act({}, ACTION_TYPES)
        assert r2 == {"type": "close_ticket"}
        r3 = agent.act({}, ACTION_TYPES)
        assert r3 == {"type": "offer_refund", "ticket_id": "t_1"}  # cycles
    finally:
        os.unlink(path)


def test_scripted_agent_falls_back_to_random_on_empty():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump([], f)
        path = f.name
    try:
        agent = ScriptedAgent(path)
        result = agent.act({}, ACTION_TYPES)
        assert result["type"] in ACTION_TYPES
    finally:
        os.unlink(path)


def test_scripted_agent_falls_back_on_invalid_action_type():
    actions = [{"type": "nonexistent_action"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(actions, f)
        path = f.name
    try:
        agent = ScriptedAgent(path)
        result = agent.act({}, ACTION_TYPES)
        assert result["type"] in ACTION_TYPES
    finally:
        os.unlink(path)


def test_make_agent_random():
    agent = make_agent("random")
    assert isinstance(agent, RandomAgent)


def test_make_agent_scripted():
    actions = [{"type": "close_ticket"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(actions, f)
        path = f.name
    try:
        agent = make_agent(f"scripted:{path}")
        assert isinstance(agent, ScriptedAgent)
    finally:
        os.unlink(path)


def test_make_agent_unknown_raises():
    with pytest.raises(ValueError, match="Unknown agent_id"):
        make_agent("unknown_type:foo")


def test_anthropic_agent_calls_api():
    from forge.runtime.agents.anthropic_agent import AnthropicAgent
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_tool_use = MagicMock()
    mock_tool_use.type = "tool_use"
    mock_tool_use.name = "offer_refund"
    mock_tool_use.input = {"ticket_id": "t_42"}
    mock_response.content = [mock_tool_use]
    mock_client.messages.create.return_value = mock_response
    agent = AnthropicAgent(model="claude-sonnet-4-6", client=mock_client)
    result = agent.act({"state": "some_obs"}, frozenset(["offer_refund"]))
    assert result == {"type": "offer_refund", "ticket_id": "t_42"}
    mock_client.messages.create.assert_called_once()


def test_openai_agent_calls_api():
    from forge.runtime.agents.openai_agent import OpenAIAgent
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "close_ticket"
    mock_tool_call.function.arguments = '{"ticket_id": "t_1"}'
    mock_choice.message.tool_calls = [mock_tool_call]
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    agent = OpenAIAgent(model="gpt-4o", client=mock_client)
    result = agent.act({"state": "some_obs"}, frozenset(["close_ticket"]))
    assert result == {"type": "close_ticket", "ticket_id": "t_1"}


def test_openai_agent_uses_an_injected_prompt_template():
    from forge.runtime.agents.openai_agent import OpenAIAgent
    from forge.runtime.prompting import ForgeAgentPromptTemplate

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.tool_calls = []
    mock_client.chat.completions.create.return_value.choices = [mock_choice]
    agent = OpenAIAgent(
        model="gpt-4o",
        client=mock_client,
        prompt_template=ForgeAgentPromptTemplate(),
    )

    with pytest.raises(AgentError):
        agent.act({"b": 2, "a": 1}, frozenset(["close_ticket"]))

    request = mock_client.chat.completions.create.call_args.kwargs
    assert request["messages"][1]["content"].find('"a": 1') < request["messages"][1]["content"].find('"b": 2')
    assert request["tools"][0]["function"]["name"] == "close_ticket"


def test_make_agent_anthropic():
    with patch("forge.runtime.agents.anthropic_agent.anthropic") as mock_anthropic:
        mock_anthropic.Anthropic.return_value = MagicMock()
        agent = make_agent("anthropic:claude-sonnet-4-6")
        from forge.runtime.agents.anthropic_agent import AnthropicAgent
        assert isinstance(agent, AnthropicAgent)


def test_make_agent_openai():
    with patch("forge.runtime.agents.openai_agent.openai") as mock_openai:
        mock_openai.OpenAI.return_value = MagicMock()
        agent = make_agent("openai:gpt-4o")
        from forge.runtime.agents.openai_agent import OpenAIAgent
        assert isinstance(agent, OpenAIAgent)


def test_make_agent_binds_environment_prompt_task_and_tool_schema():
    from forge.contracts import Task, ToolParam, ToolSpec
    from forge.runtime.prompting import ForgeAgentPromptTemplate
    from forge.runtime.task_source import StaticTaskSource
    from forge.runtime.tools import SpecToolProvider

    class _Environment:
        prompt = ForgeAgentPromptTemplate()
        task_source = StaticTaskSource([Task(id="close", objective="Close ticket 42")])
        tools = SpecToolProvider([
            ToolSpec(name="close_ticket", params=[ToolParam(name="ticket_id")])
        ])
        current_task = None

    client = MagicMock()
    choice = MagicMock()
    choice.message.tool_calls = []
    client.chat.completions.create.return_value.choices = [choice]
    agent = make_agent("openai:gpt-4o", environment=_Environment(), client=client)

    with pytest.raises(AgentError):
        agent.act({"ticket": 42}, frozenset(["close_ticket"]))

    request = client.chat.completions.create.call_args.kwargs
    assert "Close ticket 42" in request["messages"][1]["content"]
    parameters = request["tools"][0]["function"]["parameters"]
    assert parameters["required"] == ["ticket_id"]


# ---------------------------------------------------------------------------
# A model that answers without a tool call made no move. The agent must say
# so instead of inventing one, or calibration grades a step nobody chose.
# ---------------------------------------------------------------------------

def test_an_anthropic_reply_without_a_tool_call_raises_instead_of_inventing_an_action():
    from forge.runtime.agents.anthropic_agent import AnthropicAgent

    client = MagicMock()
    text = MagicMock()
    text.type = "text"
    text.text = "I think you should close the ticket."
    client.messages.create.return_value.content = [text]
    agent = AnthropicAgent(model="claude-sonnet-4-6", client=client)

    with pytest.raises(AgentError, match="no tool call"):
        agent.act({}, frozenset(["close_ticket", "escalate"]))


def test_an_openai_reply_without_a_tool_call_raises_instead_of_inventing_an_action():
    from forge.runtime.agents.openai_agent import OpenAIAgent

    client = MagicMock()
    choice = MagicMock()
    choice.message.tool_calls = None
    choice.message.content = "Done."
    client.chat.completions.create.return_value.choices = [choice]
    agent = OpenAIAgent(model="gpt-4o", client=client)

    with pytest.raises(AgentError, match="no tool call"):
        agent.act({}, frozenset(["close_ticket", "escalate"]))


def test_make_agent_ollama_uses_the_openai_compatible_endpoint(monkeypatch):
    from forge.runtime.agents.ollama_agent import OllamaAgent

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")
    with patch("forge.runtime.agents.openai_agent.openai") as mock_openai:
        agent = make_agent("ollama:qwen3.8:27b")

    assert isinstance(agent, OllamaAgent)
    assert agent._model == "qwen3.8:27b"
    kwargs = mock_openai.OpenAI.call_args.kwargs
    assert kwargs["base_url"] == "http://gpu-box:11434/v1"
    assert kwargs["api_key"] != ""


def test_make_agent_ollama_without_a_model_is_rejected():
    with pytest.raises(AgentError, match="model"):
        make_agent("ollama:")

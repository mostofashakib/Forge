from __future__ import annotations

from types import SimpleNamespace

from forge.runtime.agent_logger import AgentRunLogger
from forge.runtime.agents.anthropic_agent import AnthropicAgent
from forge.runtime.agents.openai_agent import OpenAIAgent


class _FakeAnthropic:
    def __init__(self, blocks) -> None:
        self._blocks = blocks
        self.calls: list[dict] = []

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self._blocks)


class _FakeOpenAI:
    def __init__(self, tool_calls, content=None) -> None:
        message = SimpleNamespace(tool_calls=tool_calls, content=content)
        self._response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

        create = self._create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    def _create(self, **kwargs):
        return self._response


def test_anthropic_agent_logs_llm_call_with_response():
    blocks = [SimpleNamespace(type="tool_use", name="inc", input={"n": 1})]
    logger = AgentRunLogger(run_id="ep")
    logger.start_run()
    logger.set_step(0)
    agent = AnthropicAgent(model="m", client=_FakeAnthropic(blocks), logger=logger)

    action = agent.act({"x": 0}, frozenset({"inc"}))

    assert action == {"type": "inc", "n": 1}
    llm = [e for e in logger.entries if e.kind == "llm_call"]
    assert len(llm) == 1
    assert llm[0].step == 0
    assert llm[0].payload["tool_call"] == {"type": "inc", "n": 1}
    # The raw response is captured in a serializable form.
    assert "inc" in str(llm[0].payload["response"])
    assert llm[0].payload["prompt"] is not None


def test_anthropic_agent_logs_fallback_action_when_no_tool_use():
    blocks = [SimpleNamespace(type="text", text="thinking...")]
    logger = AgentRunLogger(run_id="ep")
    logger.start_run()
    agent = AnthropicAgent(model="m", client=_FakeAnthropic(blocks), logger=logger)

    action = agent.act({"x": 0}, frozenset({"inc", "dec"}))

    assert action == {"type": "dec"}  # sorted()[0]
    assert [e.payload["tool_call"] for e in logger.entries if e.kind == "llm_call"] == [
        {"type": "dec"}
    ]


def test_anthropic_agent_without_logger_still_acts():
    blocks = [SimpleNamespace(type="tool_use", name="inc", input={})]
    agent = AnthropicAgent(model="m", client=_FakeAnthropic(blocks))
    assert agent.act({"x": 0}, frozenset({"inc"})) == {"type": "inc"}


def test_openai_agent_logs_llm_call_with_response():
    tool_calls = [SimpleNamespace(function=SimpleNamespace(name="inc", arguments='{"n": 2}'))]
    logger = AgentRunLogger(run_id="ep")
    logger.start_run()
    logger.set_step(1)
    agent = OpenAIAgent(model="m", client=_FakeOpenAI(tool_calls), logger=logger)

    action = agent.act({"x": 0}, frozenset({"inc"}))

    assert action == {"type": "inc", "n": 2}
    llm = [e for e in logger.entries if e.kind == "llm_call"]
    assert len(llm) == 1
    assert llm[0].step == 1
    assert llm[0].payload["tool_call"] == {"type": "inc", "n": 2}
    assert "inc" in str(llm[0].payload["response"])


def test_openai_agent_without_logger_still_acts():
    tool_calls = [SimpleNamespace(function=SimpleNamespace(name="inc", arguments="{}"))]
    agent = OpenAIAgent(model="m", client=_FakeOpenAI(tool_calls))
    assert agent.act({"x": 0}, frozenset({"inc"})) == {"type": "inc"}

import pytest
from forge.envgen.agents.telemetry import TelemetryAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.schemas import GeneratedApp, FileContent
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.schemas import CompilerInput


def _ctx():
    return EnvGenContext(
        env_name="ticket_env",
        description="ticket system",
        compiler_input=CompilerInput(
            project_name="ticket_env", domain="support",
            entities=[], actions=[], tasks=[],
        ),
    )


@pytest.mark.asyncio
async def test_telemetry_agent_depends_on_app_code():
    agent = TelemetryAgent(client=MockLLMClient({
        "GeneratedApp": GeneratedApp(files=[FileContent(path="main.py", content="# instrumented")])
    }))
    assert "app_code" in agent.depends_on
    assert agent.produces == "instrumented_code"


@pytest.mark.asyncio
async def test_telemetry_agent_waits_for_app_code_and_publishes():
    mock = MockLLMClient({
        "GeneratedApp": GeneratedApp(files=[
            FileContent(path="main.py", content="# instrumented with redis.xadd calls")
        ])
    })
    agent = TelemetryAgent(client=mock)
    bus = ArtifactBus()
    await bus.publish("app_code", {"main.py": "# original"})
    await agent.run(_ctx(), bus)
    result = bus.get("instrumented_code")
    assert result is not None
    assert "main.py" in result

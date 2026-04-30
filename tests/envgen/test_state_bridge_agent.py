import pytest
from forge.envgen.agents.state_bridge import StateBridgeAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.schemas import GeneratedFile
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.schemas import CompilerInput, ActionDef


def _ctx():
    return EnvGenContext(
        env_name="ticket_env",
        description="ticket system",
        compiler_input=CompilerInput(
            project_name="ticket_env", domain="support",
            entities=[], actions=[ActionDef(name="close_ticket", params=[])], tasks=[],
        ),
    )


@pytest.mark.asyncio
async def test_state_bridge_depends_on_instrumented_code():
    agent = StateBridgeAgent(client=MockLLMClient({
        "GeneratedFile": GeneratedFile(content="class ContainerForgeEnv: pass")
    }))
    assert "instrumented_code" in agent.depends_on
    assert agent.produces == "state_bridge_code"


@pytest.mark.asyncio
async def test_state_bridge_publishes_python_source():
    mock = MockLLMClient({
        "GeneratedFile": GeneratedFile(
            content="import gymnasium as gym\nclass ContainerForgeEnv(gym.Env):\n    def reset(self): pass\n    def step(self, a): pass"
        )
    })
    agent = StateBridgeAgent(client=mock)
    bus = ArtifactBus()
    await bus.publish("instrumented_code", {"main.py": "# instrumented"})
    await agent.run(_ctx(), bus)
    code = bus.get("state_bridge_code")
    assert isinstance(code, str)
    assert "ContainerForgeEnv" in code

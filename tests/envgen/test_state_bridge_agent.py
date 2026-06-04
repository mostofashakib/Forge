import pytest
from pydantic import ValidationError
from forge.envgen.agents.state_bridge import StateBridgeAgent, StateBridgeOutput
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.schemas import CompilerInput, ActionDef
from forge.schema.state_schema import StateSchemaManifest, FieldSpec


def _ctx():
    return EnvGenContext(
        env_name="ticket_env",
        description="ticket system",
        compiler_input=CompilerInput(
            project_name="ticket_env", domain="support",
            entities=[], actions=[ActionDef(name="close_ticket", params=[])], tasks=[],
        ),
    )


def _mock_output() -> StateBridgeOutput:
    manifest_dict = {
        "env_name": "ticket_env",
        "fields": {
            "ticket_count": {"type": "integer", "volatile": False, "derived_from": [], "required": True},
            "open_ticket_ids": {"type": "array", "volatile": False, "derived_from": [], "required": True},
        },
    }
    return StateBridgeOutput(
        state_bridge_code="import gymnasium as gym\nclass ContainerForgeEnv(gym.Env):\n    def reset(self): pass\n    def step(self, a): pass",
        state_schema_manifest=manifest_dict,
    )


@pytest.mark.asyncio
async def test_state_bridge_depends_on_instrumented_code():
    agent = StateBridgeAgent(client=MockLLMClient({"StateBridgeOutput": _mock_output()}))
    assert "instrumented_code" in agent.depends_on
    assert agent.produces == ["state_bridge_code", "state_schema_manifest"]


@pytest.mark.asyncio
async def test_state_bridge_publishes_python_source():
    agent = StateBridgeAgent(client=MockLLMClient({"StateBridgeOutput": _mock_output()}))
    bus = ArtifactBus()
    await bus.publish("instrumented_code", {"main.py": "# instrumented"})
    await agent.run(_ctx(), bus)
    code = bus.get("state_bridge_code")
    assert isinstance(code, str)
    assert "ContainerForgeEnv" in code


@pytest.mark.asyncio
async def test_state_bridge_publishes_manifest():
    agent = StateBridgeAgent(client=MockLLMClient({"StateBridgeOutput": _mock_output()}))
    bus = ArtifactBus()
    await bus.publish("instrumented_code", {"main.py": "# instrumented"})
    await agent.run(_ctx(), bus)
    manifest = bus.get("state_schema_manifest")
    assert isinstance(manifest, StateSchemaManifest)
    assert manifest.env_name == "ticket_env"
    assert "ticket_count" in manifest.fields


@pytest.mark.asyncio
async def test_state_bridge_manifest_invalid_json_raises():
    bad_output = StateBridgeOutput(
        state_bridge_code="class ContainerForgeEnv: pass",
        state_schema_manifest={"env_name": "x", "fields": {"f": {"type": "INVALID_TYPE"}}},
    )
    agent = StateBridgeAgent(client=MockLLMClient({"StateBridgeOutput": bad_output}))
    bus = ArtifactBus()
    await bus.publish("instrumented_code", {"main.py": ""})
    with pytest.raises(ValidationError):
        await agent.run(_ctx(), bus)

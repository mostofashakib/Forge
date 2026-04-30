import pytest
import yaml
from forge.envgen.agents.policy import PolicyAgent
from forge.envgen.agents.reward import RewardAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.schemas import GeneratedFile
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.schemas import CompilerInput


def _ctx(policy_req="", reward_req=""):
    return EnvGenContext(
        env_name="ticket_env",
        description="ticket system",
        compiler_input=CompilerInput(
            project_name="ticket_env", domain="support",
            entities=[], actions=[], tasks=[],
        ),
        policy_requirements=policy_req,
        reward_requirements=reward_req,
    )


@pytest.mark.asyncio
async def test_policy_agent_uses_default_when_no_requirements():
    agent = PolicyAgent(client=MockLLMClient({}))  # no mock needed — skips LLM
    bus = ArtifactBus()
    await agent.run(_ctx(policy_req=""), bus)
    content = bus.get("policy_dsl")
    assert content is not None
    parsed = yaml.safe_load(content)
    assert "policies" in parsed


@pytest.mark.asyncio
async def test_policy_agent_calls_llm_when_requirements_provided():
    mock = MockLLMClient({
        "GeneratedFile": GeneratedFile(content="policies:\n  - id: no_delete\n    condition: \"False\"\n    forbidden_actions: []\n    description: no delete")
    })
    agent = PolicyAgent(client=mock)
    bus = ArtifactBus()
    await agent.run(_ctx(policy_req="no deleting records"), bus)
    content = bus.get("policy_dsl")
    assert "no_delete" in content


@pytest.mark.asyncio
async def test_reward_agent_uses_default_when_no_requirements():
    agent = RewardAgent(client=MockLLMClient({}))
    bus = ArtifactBus()
    await agent.run(_ctx(reward_req=""), bus)
    code = bus.get("reward_fn_code")
    assert "compute_reward" in code
    assert "RewardBreakdown" in code


@pytest.mark.asyncio
async def test_reward_agent_calls_llm_when_requirements_provided():
    mock = MockLLMClient({
        "GeneratedFile": GeneratedFile(content="from forge.runtime.reward import RewardBreakdown, RewardComponent\ndef compute_reward(state, trajectory, verifier_results, task) -> RewardBreakdown:\n    return RewardBreakdown(total_reward=1.0, components=[])")
    })
    agent = RewardAgent(client=mock)
    bus = ArtifactBus()
    await agent.run(_ctx(reward_req="binary reward only"), bus)
    code = bus.get("reward_fn_code")
    assert "compute_reward" in code

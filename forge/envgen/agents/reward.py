from __future__ import annotations
import asyncio
from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.schemas import GeneratedFile
from forge.extraction.llm_client import LLMClient, get_client

_DEFAULT_REWARD = """\
from forge.runtime.reward import RewardBreakdown, RewardComponent


def compute_reward(state, trajectory, verifier_results, task) -> RewardBreakdown:
    components = []
    passed = any(vr.passed for vr in verifier_results)
    components.append(RewardComponent(name="task_success", value=1.0 if passed else 0.0))
    step_count = len(trajectory.steps) if trajectory and trajectory.steps else 0
    components.append(RewardComponent(name="step_penalty", value=-0.01 * step_count))
    violation_count = sum(
        1 for s in (trajectory.steps if trajectory and trajectory.steps else [])
        for e in (s.events or [])
        if isinstance(e, dict) and e.get("type") == "policy_violation"
    )
    components.append(RewardComponent(name="violation_penalty", value=-0.5 * violation_count))
    total = max(-1.0, min(1.0, sum(c.value for c in components)))
    return RewardBreakdown(total_reward=total, components=components)
"""

_SYSTEM = (
    "Generate a Python reward function for a reinforcement learning environment.\n"
    "The function MUST have this exact signature:\n"
    "  def compute_reward(state, trajectory, verifier_results, task) -> RewardBreakdown:\n"
    "Import RewardBreakdown and RewardComponent from forge.runtime.reward.\n"
    "Base the logic on the user's requirements. Return a complete Python file.\n"
    "Call the extract tool with the result."
)


class RewardAgent(EnvGenAgent):
    depends_on: list[str] = []
    produces: list[str] = ["reward_fn_code"]

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or get_client(max_tokens=2048)

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        if not ctx.reward_requirements.strip():
            await bus.publish("reward_fn_code", _DEFAULT_REWARD)
            return
        user = f"Reward requirements:\n{ctx.reward_requirements}"
        loop = asyncio.get_event_loop()
        result: GeneratedFile = await loop.run_in_executor(
            None, lambda: self._client.extract(system=_SYSTEM, user=user, schema=GeneratedFile)
        )
        await bus.publish("reward_fn_code", result.content)

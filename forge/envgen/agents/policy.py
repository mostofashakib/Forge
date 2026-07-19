from __future__ import annotations
import asyncio
from forge.envgen.agents.base import EnvGenAgent, render_correction_context, with_correction
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.schemas import GeneratedFile
from forge.extraction.llm_client import LLMClient, get_client
from forge.envgen.config import envgen_config

_DEFAULT_POLICY = """\
policies:
  - id: no_bulk_destructive
    condition: "action.get('count', 1) > 10"
    forbidden_actions: []
    description: "Prevent bulk operations affecting more than 10 items"
  - id: rate_limit_placeholder
    condition: "False"
    forbidden_actions: []
    description: "Placeholder — all actions rate-limited to 60/minute at proxy layer"
"""

_SYSTEM = (
    "Generate a policies.yaml file in the Forge PolicyEngine DSL.\n"
    "Each policy must have:\n"
    "  id: snake_case string\n"
    "  condition: Python boolean expression evaluated against action dict and state dict\n"
    "  forbidden_actions: list of action name strings (empty list if condition-based)\n"
    "  description: plain English explanation\n"
    "Base policies on the user's requirements. Return valid YAML.\n"
    "Call the extract tool with the result."
)


class PolicyPrompts:
    SYSTEM = _SYSTEM


class PolicyAgent(EnvGenAgent):
    agent_id = "policy"
    optional_depends_on: list[str] = ["rl_research"]
    produces: list[str] = ["policy_dsl"]

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or get_client(max_tokens=envgen_config().fast_llm_tokens)

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        correction = render_correction_context(bus, self.agent_id)
        if not ctx.policy_requirements.strip() and not correction:
            await bus.publish("policy_dsl", _DEFAULT_POLICY)
            return
        action_names = [a.name for a in ctx.compiler_input.actions]
        requirements = ctx.policy_requirements.strip() or (
            "(none provided; infer suitable policies from the actions and the "
            "corrections below)"
        )
        user = f"Actions: {action_names}\n\nPolicy requirements:\n{requirements}"
        research = bus.get("rl_research")
        if research is not None:
            user += f"\n\nRESEARCHED RL CONTEXT:\n{research.as_prompt()}"
        user = with_correction(bus, self.agent_id, user)
        loop = asyncio.get_event_loop()
        result: GeneratedFile = await loop.run_in_executor(
            None,
            lambda: self._client.extract(
                system=PolicyPrompts.SYSTEM, user=user, schema=GeneratedFile
            ),
        )
        await bus.publish("policy_dsl", result.content)

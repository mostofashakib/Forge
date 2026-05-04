from __future__ import annotations
import asyncio
from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.schemas import GeneratedFile
from forge.extraction.llm_client import LLMClient, get_client

_SYSTEM = (
    "Generate a ContainerForgeEnv class — a standalone gymnasium.Env subclass (do NOT extend ForgeEnv).\n"
    "The class wraps a containerized FastAPI app via HTTP using httpx (sync).\n\n"
    "Required interface:\n"
    "  def __init__(self, base_url: str) -> None:\n"
    "      self.base_url = base_url\n"
    "      self.observation_space = gymnasium.spaces.Dict({})\n"
    "      self.action_space = gymnasium.spaces.Dict({})\n\n"
    "  def reset(self, seed=None, options=None):\n"
    "      # POST {base_url}/forge/reset\n"
    "      # GET  {base_url}/forge/state → return (state_dict, {})\n\n"
    "  def step(self, action: dict):\n"
    "      # POST {base_url}/{action['type']} with action as JSON body\n"
    "      # GET  {base_url}/forge/state → obs\n"
    "      # reward = 1.0 if response status 200 else 0.0 (placeholder)\n"
    "      # terminated = False, truncated = False\n"
    "      # return (obs, reward, terminated, truncated, {})\n\n"
    "Imports needed: gymnasium, httpx\n"
    "Return the complete Python file as a single string. No placeholders.\n"
    "Call the extract tool with the result."
)


class StateBridgeAgent(EnvGenAgent):
    depends_on: list[str] = ["instrumented_code"]
    produces: str = "state_bridge_code"

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or get_client(max_tokens=4096)

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        instrumented: dict[str, str] = await bus.wait_for("instrumented_code")
        action_names = [a.name for a in ctx.compiler_input.actions]
        first_file = next(iter(instrumented.values()), "")
        user = (
            f"Action endpoint names: {action_names}\n\n"
            f"App code (first file):\n{first_file[:2000]}"
        )
        loop = asyncio.get_event_loop()
        result: GeneratedFile = await loop.run_in_executor(
            None, lambda: self._client.extract(system=_SYSTEM, user=user, schema=GeneratedFile)
        )
        await bus.publish("state_bridge_code", result.content)

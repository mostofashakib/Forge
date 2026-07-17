from __future__ import annotations
import asyncio
from pydantic import BaseModel
from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.schema.state_schema import StateSchemaManifest
from forge.extraction.llm_client import LLMClient, get_client
from forge.envgen.config import envgen_config


class StateBridgeOutput(BaseModel):
    state_bridge_code: str
    state_schema_manifest: dict


_SYSTEM = (
    "Generate two artifacts for a containerized FastAPI gymnasium environment.\n\n"
    "ARTIFACT 1 — state_bridge_code: A ContainerForgeEnv class that EXTENDS the provided base\n"
    "class (do NOT regenerate HTTP plumbing, reset, step, or observation code — the base\n"
    "already implements all of it):\n\n"
    "  from forge.envgen.container_env_base import ContainerEnvBase\n\n"
    "  class ContainerForgeEnv(ContainerEnvBase):\n"
    "      ...\n\n"
    "The base provides: __init__(base_url), reset() via POST /forge/reset + GET /forge/state,\n"
    "step(action) via POST /{action['type']} + GET /forge/state, reward 1.0 on HTTP 200 else 0.0,\n"
    "and gymnasium Dict spaces.\n\n"
    "Override ONLY when the domain requires it:\n"
    "  def action_endpoint(self, action: dict) -> str:\n"
    "      # only if an action maps to a different endpoint than /{action['type']}\n\n"
    "  def compute_reward(self, response, obs) -> float:\n"
    "      # only if the domain needs richer shaping than HTTP-200 = 1.0\n\n"
    "If no override is needed, the entire class body is `pass`.\n\n"
    "ARTIFACT 2 — state_schema_manifest: A JSON object describing every field in /forge/state.\n"
    "Format:\n"
    "  {\n"
    '    "env_name": "<name>",\n'
    '    "fields": {\n'
    '      "<field_name>": {\n'
    '        "type": "integer"|"string"|"array"|"object"|"boolean"|"datetime",\n'
    '        "volatile": true/false,  // true for timestamps, auto-increment IDs\n'
    '        "derived_from": ["<action_endpoint>", ...],  // populated only after calling this action\n'
    '        "required": true/false\n'
    "      }\n"
    "    }\n"
    "  }\n\n"
    "Include ALL fields visible in /forge/state, including derived state such as search results, "
    "active filters, selected items, and pagination cursors. Mark a field volatile=true if it "
    "changes without any user action (e.g. timestamps). List derived_from action endpoint names "
    "for fields that are only populated after calling a specific action.\n\n"
    "Return both artifacts. Call the extract tool with the result."
)


class StateBridgePrompts:
    SYSTEM = _SYSTEM


class StateBridgeAgent(EnvGenAgent):
    agent_id = "state_bridge"
    depends_on: list[str] = ["instrumented_code"]
    optional_depends_on: list[str] = ["rl_research"]
    produces: list[str] = ["state_bridge_code", "state_schema_manifest"]

    def __init__(
        self,
        client: LLMClient | None = None,
        missing_fields_feedback: list[str] | None = None,
    ) -> None:
        self._client = client or get_client(
            max_tokens=envgen_config().standard_llm_tokens
        )
        self._missing_fields_feedback = missing_fields_feedback or []

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        instrumented: dict[str, str] = await bus.wait_for("instrumented_code")
        action_names = [a.name for a in ctx.compiler_input.actions]
        first_file = next(iter(instrumented.values()), "")
        input_chars = envgen_config().state_bridge_input_chars
        user = f"Action endpoint names: {action_names}\n\nApp code (first file):\n{first_file[:input_chars]}"
        research = bus.get("rl_research")
        if research is not None:
            user += f"\n\nRESEARCHED RL CONTEXT:\n{research.as_prompt()}"
        if self._missing_fields_feedback:
            user += (
                f"\n\nPREVIOUS VALIDATION FAILED. These declared fields were not found in the "
                f"actual /forge/state response: {self._missing_fields_feedback}. "
                "Remove or correct these fields in state_schema_manifest so the manifest "
                "accurately reflects what the app actually exposes."
            )
        loop = asyncio.get_event_loop()
        result: StateBridgeOutput = await loop.run_in_executor(
            None,
            lambda: self._client.extract(
                system=StateBridgePrompts.SYSTEM,
                user=user,
                schema=StateBridgeOutput,
            ),
        )
        # Validate the manifest blob immediately — raises ValidationError on bad schema
        manifest = StateSchemaManifest.model_validate(result.state_schema_manifest)
        await bus.publish("state_bridge_code", result.state_bridge_code)
        await bus.publish("state_schema_manifest", manifest)

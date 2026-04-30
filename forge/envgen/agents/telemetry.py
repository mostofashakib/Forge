from __future__ import annotations
import asyncio
from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.schemas import GeneratedApp
from forge.extraction.llm_client import AnthropicClient, LLMClient

_SYSTEM = (
    "Instrument the provided FastAPI app to emit events to Redis Streams on every state-mutating route.\n"
    "For each POST endpoint that changes state, capture state before and after, then add:\n\n"
    "  await redis_client.xadd(f'forge:events:{env_name}', {\n"
    "      b'timestamp': datetime.now(timezone.utc).isoformat().encode(),\n"
    "      b'episode_id': request.headers.get('X-Forge-Episode', '').encode(),\n"
    "      b'actor': request.headers.get('X-Forge-Actor', 'user').encode(),\n"
    "      b'action_type': b'<endpoint_name>',\n"
    "      b'payload': json.dumps(payload_dict).encode(),\n"
    "      b'state_before': json.dumps(state_before).encode(),\n"
    "      b'state_after': json.dumps(state_after).encode(),\n"
    "  })\n\n"
    "Add at app startup: redis_client = redis.asyncio.from_url(os.environ['REDIS_URL'])\n"
    "Add missing imports: json, os, datetime, timezone, redis.asyncio\n"
    "Ensure these endpoints exist (add if missing): /forge/health, /forge/state, /forge/reset,\n"
    "  /forge/snapshot, /forge/restore/{slot}, /forge/restore-state\n"
    "Return complete modified files. No placeholders.\n"
    "Call the extract tool with all results."
)


class TelemetryAgent(EnvGenAgent):
    depends_on: list[str] = ["app_code"]
    produces: str = "instrumented_code"

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or AnthropicClient(max_tokens=32768)

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        app_code: dict[str, str] = await bus.wait_for("app_code")
        files_text = "\n\n".join(
            f"=== {path} ===\n{content}" for path, content in app_code.items()
        )
        user = (
            f"Redis stream key: forge:events:{ctx.env_name}\n"
            f"env_name variable value: {ctx.env_name}\n\n"
            f"App code:\n{files_text}"
        )
        loop = asyncio.get_event_loop()
        result: GeneratedApp = await loop.run_in_executor(
            None, lambda: self._client.extract(system=_SYSTEM, user=user, schema=GeneratedApp)
        )
        await bus.publish("instrumented_code", {f.path: f.content for f in result.files})

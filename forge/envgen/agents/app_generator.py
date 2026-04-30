from __future__ import annotations
import asyncio
from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.schemas import GeneratedApp
from forge.extraction.llm_client import AnthropicClient, LLMClient

_SYSTEM = (
    "Generate a complete, runnable FastAPI Python application that simulates the described workflow.\n"
    "Requirements:\n"
    "  1. One POST endpoint per action (e.g. POST /close_ticket), JSON body matching action params\n"
    "  2. SQLite persistence via SQLAlchemy (sync, file 'app.db'), models mirror the entity schema\n"
    "  3. Minimal HTML UI at GET /ui — shows current state and a form per action\n"
    "  4. These Forge endpoints:\n"
    "     GET  /forge/health            → {\"status\": \"ok\"}\n"
    "     GET  /forge/state             → full current state as JSON\n"
    "     POST /forge/reset             → drop all rows, re-seed initial state, return {\"ok\": true}\n"
    "     POST /forge/snapshot          → body {\"slot\": \"name\"}, save state, return {\"ok\": true}\n"
    "     POST /forge/restore/{slot}    → restore saved state, return {\"ok\": true}\n"
    "     POST /forge/restore-state     → body is full state JSON, write directly to SQLite, return {\"ok\": true}\n"
    "No placeholders. No TODOs. Return complete runnable files.\n"
    "Call the extract tool with all results."
)


class AppGeneratorAgent(EnvGenAgent):
    depends_on: list[str] = []
    produces: str = "app_code"

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or AnthropicClient(max_tokens=8192)

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        entity_summary = "\n".join(
            f"  - {e.name}: fields={[f.name for f in e.fields]}"
            for e in ctx.compiler_input.entities
        )
        action_summary = "\n".join(
            f"  - {a.name}(params={[p.name for p in a.params]})"
            for a in ctx.compiler_input.actions
        )
        user = (
            f"Description: {ctx.description}\n"
            f"Domain: {ctx.compiler_input.domain}\n\n"
            f"Entities:\n{entity_summary}\n\n"
            f"Actions:\n{action_summary}"
        )
        loop = asyncio.get_event_loop()
        result: GeneratedApp = await loop.run_in_executor(
            None, lambda: self._client.extract(system=_SYSTEM, user=user, schema=GeneratedApp)
        )
        await bus.publish("app_code", {f.path: f.content for f in result.files})

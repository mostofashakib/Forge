from __future__ import annotations
import asyncio
from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.schemas import AppPlan, GeneratedFile
from forge.extraction.llm_client import AnthropicClient, LLMClient

_PLAN_SYSTEM = (
    "Plan the file structure for a complete, runnable FastAPI Python application.\n"
    "List only the files needed — no more. Each file has one focused responsibility.\n"
    "Typical structure: main.py, models.py, database.py, requirements.txt, Dockerfile.\n"
    "Add additional files only if the domain genuinely requires it.\n"
    "Call the extract tool with the plan."
)

_IMPL_SYSTEM = (
    "Generate the COMPLETE content for ONE file of a FastAPI Python application.\n"
    "Write only the requested file. No other files.\n"
    "Requirements:\n"
    "  - SQLite persistence via SQLAlchemy (sync, file 'app.db'), models mirror the entity schema\n"
    "  - One POST endpoint per action (e.g. POST /close_ticket), JSON body matching action params\n"
    "  - Minimal HTML UI at GET /ui — shows current state and a form per action\n"
    "  - These Forge endpoints (in main.py):\n"
    "      GET  /forge/health         → {\"status\": \"ok\"}\n"
    "      GET  /forge/state          → full current state as JSON\n"
    "      POST /forge/reset          → drop all rows, re-seed initial state, return {\"ok\": true}\n"
    "      POST /forge/snapshot       → body {\"slot\": \"name\"}, save state, return {\"ok\": true}\n"
    "      POST /forge/restore/{slot} → restore saved state, return {\"ok\": true}\n"
    "      POST /forge/restore-state  → body is full state JSON, write directly to SQLite, return {\"ok\": true}\n"
    "No placeholders. No TODOs. Complete runnable content only.\n"
    "Call the extract tool with the result."
)


class AppGeneratorAgent(EnvGenAgent):
    depends_on: list[str] = []
    produces: str = "app_code"

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or AnthropicClient(max_tokens=4096)

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        entity_summary = "\n".join(
            f"  - {e.name}: fields={[f.name for f in e.fields]}"
            for e in ctx.compiler_input.entities
        )
        action_summary = "\n".join(
            f"  - {a.name}(params={[p.name for p in a.params]})"
            for a in ctx.compiler_input.actions
        )
        app_context = (
            f"Description: {ctx.description}\n"
            f"Domain: {ctx.compiler_input.domain}\n\n"
            f"Entities:\n{entity_summary}\n\n"
            f"Actions:\n{action_summary}"
        )

        loop = asyncio.get_event_loop()

        # Phase 1: plan the file structure
        await bus.log("[app-gen] Planning file structure…")
        plan: AppPlan = await loop.run_in_executor(
            None,
            lambda: self._client.extract(
                system=_PLAN_SYSTEM,
                user=app_context,
                schema=AppPlan,
            ),
        )
        file_list = ", ".join(f.path for f in plan.files)
        await bus.log(f"[app-gen] Plan ready — {len(plan.files)} files: {file_list}")

        plan_summary = "\n".join(
            f"  - {f.path}: {f.description}" for f in plan.files
        )

        # Phase 2: generate each file individually
        files: dict[str, str] = {}
        total = len(plan.files)
        for i, file_plan in enumerate(plan.files, 1):
            await bus.log(f"[app-gen] Writing {file_plan.path} ({i}/{total})…")
            user = (
                f"{app_context}\n\n"
                f"Application file plan:\n{plan_summary}\n\n"
                f"Generate file: {file_plan.path}\n"
                f"Responsibility: {file_plan.description}"
            )
            result: GeneratedFile = await loop.run_in_executor(
                None,
                lambda fp=file_plan, u=user: self._client.extract(
                    system=_IMPL_SYSTEM,
                    user=u,
                    schema=GeneratedFile,
                ),
            )
            files[file_plan.path] = result.content
            await bus.log(f"[app-gen] {file_plan.path} ✓")

        await bus.publish("app_code", files)

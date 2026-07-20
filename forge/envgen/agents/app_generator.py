from __future__ import annotations
import asyncio
import time
from forge.envgen.agents.base import EnvGenAgent, with_correction
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.schemas import AppPlan, GeneratedFile
from forge.envgen.config import envgen_config
from forge.extraction.llm_client import LLMClient, get_client

# ---------------------------------------------------------------------------
# Plan prompt
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = (
    "Plan the file structure for a complete, runnable FastAPI Python application.\n"
    "List only the files needed — no more. Each file has one focused responsibility.\n"
    "Required files: main.py, ui.html, requirements.txt, Dockerfile.\n"
    "  main.py          — FastAPI backend: SQLAlchemy models, API routes, Forge endpoints.\n"
    "                     The /ui endpoint just serves ui.html via FileResponse.\n"
    "  ui.html          — Standalone single-page frontend (HTML + CSS + JS). No Python here.\n"
    "  requirements.txt — MUST include: fastapi, uvicorn[standard], sqlalchemy, redis,\n"
    "                     httpx, python-multipart, pydantic. The Forge build pipeline\n"
    "                     adds any of these that you forget, but listing them keeps\n"
    "                     the file honest.\n"
    "  Dockerfile       — Always FROM python:3.12-slim, EXPOSE 8000, CMD on port 8000.\n"
    "Add models.py / database.py / etc. only if the domain genuinely requires it.\n"
    "Call the extract tool with the plan."
)

_BACKEND_PLAN_SYSTEM = (
    "Plan the backend file structure for a complete, runnable FastAPI application.\n"
    "The UI is built by a separate specialist: do not include ui.html or any frontend file.\n"
    "List only files with one focused backend responsibility.\n"
    "Required files: main.py, requirements.txt, Dockerfile.\n"
    "Use SQLite with SQLAlchemy and include all required Forge endpoints.\n"
    "Docker must serve the application on port 8000.\n"
    "Call the extract tool with the plan."
)

# ---------------------------------------------------------------------------
# Implementation prompts — one per concern
# ---------------------------------------------------------------------------

_BACKEND_SYSTEM = (
    "Generate the COMPLETE content for ONE Python file of a FastAPI application.\n"
    "Write ONLY Python — no HTML, CSS, or JavaScript.\n"
    "\n"
    "REQUIREMENTS:\n"
    "  - SQLite persistence via SQLAlchemy (sync, file 'app.db'); models mirror the entity schema\n"
    "  - One POST endpoint per action (e.g. POST /close_ticket), JSON body matching action params\n"
    "  - Seed the database with realistic initial data on first startup\n"
    "  - TYPED RETURNS — mandatory: every action/tool endpoint returns a consistent, typed\n"
    "    dict with a defined shape, NEVER a bare string. Success returns a dict\n"
    "    (e.g. {\"ok\": true, ...}); errors are typed dicts too (e.g. {\"ok\": false,\n"
    "    \"error\": \"<reason>\"}) — never a bare string or f-string. The agent LLM parses\n"
    "    structured returns far more reliably than prose.\n"
    "  - Required Forge endpoints:\n"
    "      GET  /forge/health         → {\"status\": \"ok\"}\n"
    "      GET  /forge/state          → full current state as JSON\n"
    "      POST /forge/reset          → delegate to STATE.reset_state(); return {\"ok\": true}\n"
    "      POST /forge/snapshot       → body {\"slot\": \"name\"}, save state, return {\"ok\": true}\n"
    "      POST /forge/restore/{slot} → restore saved state, return {\"ok\": true}\n"
    "      POST /forge/restore-state  → body is full state JSON, write to SQLite, return {\"ok\": true}\n"
    "  - GET /ui → return FileResponse('ui.html', media_type='text/html')\n"
    "  - Add CORS middleware (allow all origins) and mount StaticFiles if needed\n"
    "  - Import FileResponse from fastapi.responses\n"
    "  - PORT IS FIXED: if you include `if __name__ == \"__main__\": uvicorn.run(...)`,\n"
    "    bind to host=\"0.0.0.0\" and port=8000. Forge always publishes 8000/tcp from\n"
    "    the container; any other port leaves the iframe with no working route.\n"
    "\n"
    "STATE-MANAGEMENT CLASS — mandatory, no exceptions:\n"
    "  Centralize ALL state in ONE state-management class (e.g. `EnvState`) instead of\n"
    "  scattering DB/query logic across endpoints. Instantiate one module-global `STATE`.\n"
    "  Explicitly define the entities, their fields/types, and their relationships in this\n"
    "  one place so it aligns with the state_schema manifest. The class MUST expose:\n"
    "      def reset_state(self) -> None:\n"
    "          # restore a fresh, reproducible universe (see determinism contract)\n"
    "      def seed_state(self, seed: int) -> None:\n"
    "          # deterministically build the initial universe from `seed`\n"
    "  Wire POST /forge/reset to STATE.reset_state(). Endpoints call methods on STATE;\n"
    "  they do not run ad-hoc queries of their own.\n"
    "\n"
    "DETERMINISM CONTRACT — mandatory, no exceptions:\n"
    "  The environment must be perfectly reproducible. NEVER call datetime.now,\n"
    "  datetime.utcnow, date.today, time.time/monotonic/perf_counter, uuid.*,\n"
    "  os.urandom, secrets.*, or unseeded random.* for any persisted state.\n"
    "  - Logical clock: define a module-global `_FORGE_CLOCK = 0` and a helper\n"
    "        def forge_now() -> int:\n"
    "            global _FORGE_CLOCK\n"
    "            value = _FORGE_CLOCK\n"
    "            _FORGE_CLOCK += 1\n"
    "            return value\n"
    "    Use forge_now() for every created_at/updated_at-style field (store the\n"
    "    integer counter, not a wall-clock timestamp).\n"
    "  - Sequential ids: define `_ID_COUNTERS: dict[str, int] = {}` and\n"
    "        def _next_id(entity: str) -> int:\n"
    "            _ID_COUNTERS[entity] = _ID_COUNTERS.get(entity, 0) + 1\n"
    "            return _ID_COUNTERS[entity]\n"
    "    Use _next_id('<entity>') for every row id. Never use uuid or random ids.\n"
    "  - STATE.reset_state() MUST first set `_FORGE_CLOCK = 0` and `_ID_COUNTERS.clear()`,\n"
    "    then re-seed (via seed_state), so a reset re-initialize()s the universe to a\n"
    "    byte-identical initial state (same rows, same ids, same counters) every time.\n"
    "\n"
    "No placeholders. No TODOs. Complete runnable Python only.\n"
    "Call the extract tool with the result."
)

_HTML_CSS_SYSTEM = (
    "Generate a COMPLETE HTML file for a production-quality single-page web application.\n"
    "Output the FULL HTML document from <!DOCTYPE html> to </html>.\n"
    "Include ALL CSS inline in a <style> block. Leave the <script> block at the bottom EMPTY\n"
    "(write exactly: <script id=\"app-js\"></script>) — JavaScript is generated separately.\n"
    "\n"
    "DESIGN MANDATE — every rule is non-negotiable:\n"
    "\n"
    "1. AESTHETIC DIRECTION\n"
    "   Commit to one bold visual direction that matches the domain:\n"
    "     brutally minimal | maximalist | retro-futuristic | luxury/refined | editorial/magazine\n"
    "     | brutalist/raw | art-deco/geometric | industrial/utilitarian | playful/toy-like\n"
    "   The design must feel like the REAL application — not a generic dev tool.\n"
    "   One look should tell a user exactly what the app does.\n"
    "\n"
    "2. TYPOGRAPHY\n"
    "   Import exactly two Google Fonts via CDN link: one display/heading, one body.\n"
    "   NEVER use Inter, Roboto, Arial, Helvetica, or system-ui — they produce generic AI slop.\n"
    "   Example pairings: DM Serif Display + DM Mono | Playfair Display + Lato |\n"
    "   Syne + IBM Plex Sans | Space Grotesk + Fira Code | Bebas Neue + Nunito.\n"
    "\n"
    "3. COLOR PALETTE\n"
    "   Define all colors in :root as CSS custom properties:\n"
    "     --bg, --surface, --surface-2, --border, --text, --text-muted,\n"
    "     --accent, --accent-hover, --danger, --success, --warning\n"
    "   Apply 60/30/10 rule: background 60 %, surface elements 30 %, accent 10 %.\n"
    "   Use a dominant dark OR light base — vary the theme; never default to white.\n"
    "   BANNED: purple gradients on white, default Bootstrap blue, flat grey on white.\n"
    "\n"
    "4. LAYOUT\n"
    "   App shell: fixed left sidebar (nav) + top bar + main content area.\n"
    "   Sidebar: app logo/name at top, nav items for each entity group, icons from Lucide CDN.\n"
    "   Top bar: page title (breadcrumb), action buttons for the active view.\n"
    "   Content: data table OR card grid (whichever fits the domain) with a toolbar.\n"
    "   Do NOT render a flat list of HTML forms — group actions as buttons on rows/cards.\n"
    "\n"
    "5. ICONS\n"
    "   Load Lucide via CDN: <script src=\"https://unpkg.com/lucide@latest\"></script>\n"
    "   Use <i data-lucide=\"icon-name\" class=\"icon\"></i> everywhere meaningful.\n"
    "   Call lucide.createIcons() in the empty script block via a DOMContentLoaded listener.\n"
    "\n"
    "6. MOTION\n"
    "   CSS transitions on: row/card hover, button :active press, sidebar item hover.\n"
    "   A @keyframes fade-slide-in for toast notifications.\n"
    "   Smooth sidebar collapse (if applicable). Nothing jarring.\n"
    "\n"
    "7. MODALS\n"
    "   One reusable modal overlay structure in the HTML (hidden by default).\n"
    "   It must have: modal-title, modal-body (form fields go here), submit button, close button.\n"
    "   Style it with a dark overlay backdrop and a well-padded surface panel.\n"
    "\n"
    "8. TOASTS\n"
    "   A #toast-container fixed to the top-right. Individual toasts styled for\n"
    "   success (var(--success)), error (var(--danger)), info (var(--accent)).\n"
    "   They stack, animate in, and animate out.\n"
    "\n"
    "9. DATA CONTAINERS\n"
    "   Provide empty but fully styled containers (table > thead/tbody or .card-grid div)\n"
    "   for every entity. JavaScript will populate them. Include a loading skeleton or spinner\n"
    "   inside each container that shows until data loads.\n"
    "\n"
    "No placeholder text. No lorem ipsum. Complete, immediately usable HTML+CSS.\n"
    "Call the extract tool with the result."
)

_JS_SYSTEM = (
    "Generate the COMPLETE JavaScript for a single-page web application.\n"
    "Output ONLY the JavaScript — no HTML, no CSS, no <script> tags.\n"
    "\n"
    "The HTML structure and all CSS already exist. Your JS will be injected into\n"
    "the empty <script id=\"app-js\"> block. You have access to the HTML structure\n"
    "and app context provided below.\n"
    "\n"
    "REQUIREMENTS:\n"
    "\n"
    "1. INITIALISATION\n"
    "   On DOMContentLoaded:\n"
    "     - Call lucide.createIcons()\n"
    "     - Load initial state: GET /forge/state\n"
    "     - Render all entity lists/tables from the state\n"
    "     - Wire up all navigation, modal, and form event listeners\n"
    "\n"
    "2. DATA RENDERING\n"
    "   Write a renderXxx(data) function for EACH entity type.\n"
    "   Clear and repopulate the relevant DOM container on every call.\n"
    "   Show a row count / summary in the sidebar nav item badge.\n"
    "   Handle empty arrays gracefully (show an 'empty state' message).\n"
    "\n"
    "3. ACTION HANDLERS\n"
    "   For EACH action:\n"
    "     - openModal(actionName, prefillData) — populate modal title + form inputs\n"
    "     - submitAction(endpoint, payload) — POST with fetch(), disable submit during flight\n"
    "     - On success: close modal, show success toast, refresh affected entity render\n"
    "     - On error: show error toast with the server's detail message\n"
    "\n"
    "4. MODAL SYSTEM\n"
    "   openModal(title, fields, onSubmit) — generic function:\n"
    "     fields = [{name, label, type, placeholder, value}]\n"
    "   Close on X click, backdrop click, or Escape key.\n"
    "\n"
    "5. TOAST SYSTEM\n"
    "   showToast(message, type) — type is 'success' | 'error' | 'info'\n"
    "   Auto-dismiss after 3 000 ms. Animate in and out.\n"
    "\n"
    "6. NAVIGATION\n"
    "   Clicking a sidebar nav item shows/hides the relevant content section.\n"
    "   Update the top-bar title and active nav state.\n"
    "\n"
    "7. LOADING STATES\n"
    "   Show a spinner / skeleton in each container while data is loading.\n"
    "   Disable action buttons while a fetch is in-flight.\n"
    "\n"
    "8. INLINE FORMATTING\n"
    "   Format dates (ISO → locale string), booleans (badge), status fields (colour badge).\n"
    "   Never render raw JSON objects or [object Object] in the UI.\n"
    "\n"
    "Write clean, modular vanilla JS. No jQuery. No frameworks. No import statements.\n"
    "No placeholders. No TODOs. Complete and immediately runnable.\n"
    "Call the extract tool with the result."
)

_GENERIC_SYSTEM = (
    "Generate the COMPLETE content for ONE file of a FastAPI Python application.\n"
    "Write only the requested file. No other files.\n"
    "No placeholders. No TODOs. Complete runnable content only.\n"
    "Call the extract tool with the result."
)

_DOCKERFILE_SYSTEM = (
    "Generate a Dockerfile for a FastAPI Python application.\n"
    "\n"
    "MANDATORY structure — these are non-negotiable so the runtime can\n"
    "publish port 8000 to the host correctly. The Forge build pipeline\n"
    "rewrites the FROM line and any port directives anyway, but match this\n"
    "exactly to keep behaviour predictable:\n"
    "\n"
    "  FROM python:3.12-slim\n"
    "  WORKDIR /app\n"
    "  COPY requirements.txt .\n"
    "  RUN pip install --no-cache-dir -r requirements.txt\n"
    "  COPY . .\n"
    "  EXPOSE 8000\n"
    "  CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n"
    "\n"
    "Add `RUN apt-get install …` lines only if the app genuinely needs system\n"
    "packages (e.g. libpq-dev for psycopg). Otherwise output exactly the seven\n"
    "lines above. Never use a port other than 8000.\n"
    "\n"
    "Output ONLY the Dockerfile contents — no markdown fences, no commentary.\n"
    "Call the extract tool with the result."
)

# Files simple enough for Haiku
_FAST_FILES = {"requirements.txt", "dockerfile"}


class AppGeneratorPrompts:
    """Prompt catalog for backend and UI generation."""

    PLAN = _PLAN_SYSTEM
    BACKEND_PLAN = _BACKEND_PLAN_SYSTEM
    BACKEND = _BACKEND_SYSTEM
    HTML_CSS = _HTML_CSS_SYSTEM
    JAVASCRIPT = _JS_SYSTEM
    GENERIC_FILE = _GENERIC_SYSTEM
    DOCKERFILE = _DOCKERFILE_SYSTEM


def _fmt(seconds: float) -> str:
    """Format elapsed seconds as '4.2s' or '1m 23s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


class BackendBuilderAgent(EnvGenAgent):
    agent_id = "backend_builder"
    optional_depends_on: list[str] = ["backend_research"]
    produces: list[str] = ["backend_code"]

    def __init__(self, client: LLMClient | None = None) -> None:
        config = envgen_config()
        # Capable tier for complex backend files such as main.py.
        self._client = client or get_client(
            max_tokens=config.capable_llm_tokens, capable=True
        )
        # Fast tier for simple template-like files (requirements.txt, Dockerfile, etc.)
        self._fast_client = client or get_client(max_tokens=config.fast_llm_tokens)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call(self, system: str, user: str, fast: bool = False) -> str:
        client = self._fast_client if fast else self._client
        loop = asyncio.get_event_loop()
        result: GeneratedFile = await loop.run_in_executor(
            None,
            lambda: client.extract(system=system, user=user, schema=GeneratedFile),
        )
        return result.content

    # ------------------------------------------------------------------
    # run — plan first, then generate all files in parallel
    # ------------------------------------------------------------------

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        run_start = time.monotonic()

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
        research = bus.get("backend_research")
        if research is not None:
            app_context += f"\n\nRESEARCHED PRODUCT CONTEXT:\n{research.as_prompt()}"
        app_context = with_correction(bus, self.agent_id, app_context)

        await bus.log(
            f"[backend-builder] Starting — "
            f"{len(ctx.compiler_input.entities)} entities, "
            f"{len(ctx.compiler_input.actions)} actions"
        )

        loop = asyncio.get_event_loop()

        # Phase 1: plan (sequential — everything depends on it)
        t_plan = time.monotonic()
        await bus.log("[backend-builder] Phase 1/2: planning backend files (sonnet)…")
        plan: AppPlan = await loop.run_in_executor(
            None,
            lambda: self._client.extract(
                system=AppGeneratorPrompts.BACKEND_PLAN,
                user=app_context,
                schema=AppPlan,
            ),
        )
        backend_files = [f for f in plan.files if f.path.lower() != "ui.html"]
        file_list = ", ".join(f.path for f in backend_files)
        await bus.log(
            f"[backend-builder] Plan done ({_fmt(time.monotonic() - t_plan)}) — "
            f"{len(backend_files)} files: {file_list}"
        )

        plan_summary = "\n".join(
            f"  - {f.path}: {f.description}" for f in backend_files
        )
        total = len(backend_files)

        # Phase 2: generate all backend files in parallel.
        await bus.log(
            f"[backend-builder] Phase 2/2: generating {total} files in parallel…"
        )
        t_gen = time.monotonic()

        async def _gen_one(i: int, file_plan) -> tuple[str, str]:
            is_simple = file_plan.path.lower() in _FAST_FILES
            model_label = "haiku" if is_simple else "sonnet"
            t_file = time.monotonic()
            await bus.log(
                f"[backend-builder] [{i}/{total}] {file_plan.path} — starting ({model_label})…"
            )

            if file_plan.path == "main.py":
                user = (
                    f"{app_context}\n\n"
                    f"Application file plan:\n{plan_summary}\n\n"
                    f"Generate file: main.py\n"
                    f"Responsibility: {file_plan.description}"
                )
                content = await self._call(system=AppGeneratorPrompts.BACKEND, user=user)

            elif file_plan.path.lower() == "dockerfile":
                # Dockerfile uses a dedicated prompt that pins port 8000;
                # post-build normalisation (`_normalise_dockerfile_port`)
                # acts as a hard guardrail in case the LLM still drifts.
                user = (
                    f"{app_context}\n\n"
                    f"Application file plan:\n{plan_summary}\n\n"
                    f"Generate file: Dockerfile\n"
                    f"Responsibility: {file_plan.description}"
                )
                content = await self._call(
                    system=AppGeneratorPrompts.DOCKERFILE, user=user, fast=True
                )

            else:
                user = (
                    f"{app_context}\n\n"
                    f"Application file plan:\n{plan_summary}\n\n"
                    f"Generate file: {file_plan.path}\n"
                    f"Responsibility: {file_plan.description}"
                )
                content = await self._call(
                    system=AppGeneratorPrompts.GENERIC_FILE, user=user, fast=is_simple
                )

            elapsed = _fmt(time.monotonic() - t_file)
            await bus.log(
                f"[backend-builder] [{i}/{total}] {file_plan.path} ✓  "
                f"({elapsed}, {len(content):,} chars)"
            )
            return file_plan.path, content

        results = await asyncio.gather(
            *[_gen_one(i, fp) for i, fp in enumerate(backend_files, 1)]
        )

        files = dict(results)
        total_chars = sum(len(v) for v in files.values())
        await bus.log(
            f"[backend-builder] All files done — "
            f"parallel wall time {_fmt(time.monotonic() - t_gen)}, "
            f"total {total_chars:,} chars across {len(files)} files"
        )
        await bus.log(
            f"[backend-builder] Total agent time: {_fmt(time.monotonic() - run_start)}"
        )

        await bus.publish("backend_code", files)


class UIBuilderAgent(EnvGenAgent):
    """Builds only the user-facing HTML, CSS, and JavaScript."""

    agent_id = "ui_builder"
    optional_depends_on: list[str] = ["ui_research"]
    produces: list[str] = ["ui_code"]

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or get_client(
            max_tokens=envgen_config().capable_llm_tokens, capable=True
        )

    async def _call(self, system: str, user: str) -> str:
        loop = asyncio.get_running_loop()
        result: GeneratedFile = await loop.run_in_executor(
            None,
            lambda: self._client.extract(system=system, user=user, schema=GeneratedFile),
        )
        return result.content

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        entity_summary = "\n".join(
            f"  - {entity.name}: fields={[field.name for field in entity.fields]}"
            for entity in ctx.compiler_input.entities
        )
        action_summary = "\n".join(
            f"  - {action.name}(params={[param.name for param in action.params]})"
            for action in ctx.compiler_input.actions
        )
        app_context = (
            f"User request: {ctx.description}\n"
            f"Domain: {ctx.compiler_input.domain}\n\n"
            f"Entities:\n{entity_summary}\n\nActions:\n{action_summary}"
        )
        research = bus.get("ui_research")
        if research is not None:
            app_context += f"\n\nRESEARCHED PRODUCT CONTEXT:\n{research.as_prompt()}"
        app_context = with_correction(bus, self.agent_id, app_context)

        await bus.log("[ui-builder] Pass 1/2: HTML structure and CSS…")
        html_css = await self._call(AppGeneratorPrompts.HTML_CSS, app_context)
        await bus.log("[ui-builder] Pass 2/2: client behavior…")
        javascript = await self._call(
            AppGeneratorPrompts.JAVASCRIPT,
            f"{app_context}\n\nHTML structure:\n---\n{html_css}\n---",
        )
        marker = '<script id="app-js"></script>'
        if marker in html_css:
            complete_html = html_css.replace(
                marker, f'<script id="app-js">\n{javascript}\n</script>'
            )
        elif "</body>" in html_css:
            complete_html = html_css.replace(
                "</body>", f"<script>\n{javascript}\n</script>\n</body>"
            )
        else:
            complete_html = f"{html_css}\n<script>\n{javascript}\n</script>"
        await bus.publish("ui_code", {"ui.html": complete_html})


class AppAssemblyAgent(EnvGenAgent):
    agent_id = "app_assembler"
    depends_on = ["backend_code", "ui_code"]
    produces = ["app_code"]

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        del ctx
        backend_code = await bus.wait_for("backend_code")
        ui_code = await bus.wait_for("ui_code")
        overlap = set(backend_code) & set(ui_code)
        if overlap:
            raise ValueError(f"Backend and UI produced the same paths: {sorted(overlap)}")
        await bus.publish("app_code", {**backend_code, **ui_code})


class AppGeneratorAgent(EnvGenAgent):
    """Compatibility facade for callers that still expect one app generator."""

    agent_id = "app_generator"
    depends_on: list[str] = []
    produces: list[str] = ["app_code"]

    def __init__(self, client: LLMClient | None = None) -> None:
        self._backend = BackendBuilderAgent(client=client)
        self._ui = UIBuilderAgent(client=client)

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        private_bus = ArtifactBus()
        await asyncio.gather(
            self._backend.run(ctx, private_bus),
            self._ui.run(ctx, private_bus),
        )
        backend_code = private_bus.get("backend_code", {})
        ui_code = private_bus.get("ui_code", {})
        await bus.publish("app_code", {**backend_code, **ui_code})

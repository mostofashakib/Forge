"""Celery tasks that build sandbox environments and retire expired ones."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from backend.app.worker.celery_app import celery
from backend.app.worker.env_artifacts import load_manifest
from forge.settings import generated_envs_root, redis_url

logger = logging.getLogger("backend.app.worker.tasks")


def _write_personas(env_dir, personas: dict) -> int:
    """Persist the cast chosen in the builder into the environment's config.

    The envgen pipeline never creates a `custom/` directory — only the compiler
    does — so this creates one. Nobody is given an action here: the app's
    endpoints exist only now that it is generated, and binding them is a
    decision the author makes on the personas page against the real surface.
    A failure to write is logged and swallowed: losing the cast is worth far
    less than losing a build that otherwise succeeded.
    """
    import yaml

    from forge.personas.config import dump_population, load_population

    try:
        population = load_population(personas)
        custom_dir = env_dir / "custom"
        custom_dir.mkdir(parents=True, exist_ok=True)
        config_path = custom_dir / "config.yaml"
        raw = {}
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text()) or {}
        raw["personas"] = dump_population(population)
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
        return len(population.roster) + len(population.archetypes)
    except Exception as exc:
        logger.warning("[personas] could not write cast for %s: %s", env_dir.name, exc)
        return 0

def pipeline_flags(plan) -> dict[str, bool]:
    """Which optional specialists a generation plan actually includes.

    Streamed to the progress UI so it renders a checklist matching the build
    that is really running, rather than a fixed list of every possible agent.
    """
    agent_ids = {task.agent_id for task in plan.tasks}
    return {
        "user_researcher_enabled": "user_researcher" in agent_ids,
        "ui_builder_enabled": "ui_builder" in agent_ids,
    }

_ARTIFACT_LABELS = {
    "generation_plan":   "Prompt Planner",
    "backend_research":  "User Research (backend context)",
    "ui_research":       "User Research (UI context)",
    "rl_research":       "User Research (RL context)",
    "reviewer_research": "User Research (review context)",
    "backend_code":      "Backend Builder",
    "ui_code":           "UI Builder",
    "app_code":          "App Assembly",
    "instrumented_code": "Telemetry Instrumentation",
    "state_bridge_code": "State Bridge",
    "policy_dsl":        "Policy Rules",
    "reward_fn_code":    "Reward Function",
    "correctness_report": "Correctness Reviewer",
    "review_report":     "Quality Reviewer",
}

async def _check_correctness(base_url: str, action_names: list[str], publish) -> None:
    """Prove reset fidelity and snapshot/restore on the live container.

    Raises CorrectnessValidationError when the environment is broken. A gate
    that cannot reach the container is reported and does not fail the build.
    """
    from forge.envgen.correctness_validator import (
        CorrectnessValidationError, CorrectnessValidator,
    )

    publish({"log": "[forge] validating reset fidelity and snapshot/restore…"})
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None, lambda: CorrectnessValidator(base_url=base_url).validate(action_names)
        )
    except Exception as exc:  # container not ready / transport error
        publish({"log": f"[forge] correctness validation could not run: {exc}"})
        return
    if not result.passed:
        for finding in result.findings:
            publish({"log": f"[forge] correctness FAIL [{finding.category}]: {finding.message}"})
        raise CorrectnessValidationError(result)
    publish({"log": "[forge] correctness validation passed ✓"})

async def _validate_manifest(
    *,
    env_name: str,
    description: str,
    compiler_input,
    base_url: str,
    env_dir: Path,
    publish,
    max_attempts: int = 3,
) -> None:
    """Check the state manifest against the live app, re-running the state
    bridge with the missing fields as feedback until it passes or gives up.
    """
    from forge.envgen.agents.state_bridge import StateBridgeAgent
    from forge.envgen.artifact_bus import ArtifactBus
    from forge.envgen.context import EnvGenContext
    from forge.envgen.post_generation_validator import PostGenerationValidator

    manifest = load_manifest(env_dir)
    if manifest is None:
        return
    manifest_path = env_dir / "state_schema.json"
    app_dir = env_dir / "app"
    loop = asyncio.get_running_loop()

    for attempt in range(max_attempts):
        publish({"log": f"[forge] validating manifest (attempt {attempt + 1}/{max_attempts})…"})
        try:
            result = await loop.run_in_executor(
                None,
                lambda m=manifest: PostGenerationValidator(base_url=base_url).validate(m),
            )
        except Exception as exc:
            publish({"log": f"[forge] manifest validation error (container not ready?): {exc}"})
            return
        if result.passed:
            publish({"log": f"[forge] manifest validation passed (coverage={result.coverage_score:.2f}) ✓"})
            _update_sandbox(env_name, state_schema=manifest.model_dump_json())
            return
        publish({"log": f"[forge] manifest validation failed — missing fields: {result.missing_fields}"})
        if attempt == max_attempts - 1:
            _update_sandbox(env_name, validation_missing_fields=json.dumps(result.missing_fields))
            publish({
                "log": f"[forge] WARNING: manifest validation gave up after "
                       f"{max_attempts} attempts. Missing: {result.missing_fields}"
            })
            return

        publish({"log": "[forge] re-running state bridge agent with missing field feedback…"})
        ctx = EnvGenContext(env_name=env_name, description=description, compiler_input=compiler_input)
        retry_bus = ArtifactBus()
        # The state bridge reads the instrumented app, so load it back from disk.
        instrumented = (
            {str(p.relative_to(app_dir)): p.read_text() for p in app_dir.rglob("*.py")}
            if app_dir.exists() else {}
        )
        await retry_bus.publish("instrumented_code", instrumented)
        await StateBridgeAgent(missing_fields_feedback=result.missing_fields).run(ctx, retry_bus)
        new_manifest = retry_bus.get("state_schema_manifest")
        if new_manifest is not None:
            manifest = new_manifest
            manifest_path.write_text(manifest.model_dump_json())
            publish({"log": "[forge] state bridge agent produced updated manifest ✓"})
        new_bridge = retry_bus.get("state_bridge_code")
        if new_bridge:
            (env_dir / "container_env.py").write_text(new_bridge)

def _update_sandbox(env_name: str, **fields) -> None:
    from backend.app.database import get_session_factory
    from backend.app.models import SandboxEnvironment

    with get_session_factory()() as db:
        sandbox = db.get(SandboxEnvironment, env_name)
        if sandbox:
            for name, value in fields.items():
                setattr(sandbox, name, value)
            db.commit()

@dataclass(frozen=True)
class BuildRequest:
    """What the builder asked for. `env_type` picks the build path."""

    env_name: str
    env_type: str = "general"
    description: str = ""
    domain: str = "localhost"
    policy_requirements: str = ""
    reward_requirements: str = ""
    reference_urls: tuple[str, ...] = ()
    use_user_researcher: bool = False
    with_ui: bool = False
    source_product_name: str = ""
    source_product_url: str = ""
    personas: dict | None = None


class BuildReporter:
    """Streams build progress to the UI and records the sandbox's status."""

    def __init__(self, env_name: str, publish: Callable[[dict], None]) -> None:
        self._env_name = env_name
        self.publish = publish

    def log(self, message: str) -> None:
        self.publish({"log": message})

    def building(self) -> None:
        _update_sandbox(self._env_name, status="building")

    def running(self, container_id: str, port: int, image_tag: str) -> None:
        _update_sandbox(
            self._env_name,
            status="running",
            container_id=container_id,
            container_port=port or None,
            image_tag=image_tag,
        )

    def failed(self) -> None:
        # Clear container/image references too — otherwise a leftover tag from
        # a previous successful build stays in the DB, and /start would later
        # try to spin up a container against an image that may no longer exist.
        _update_sandbox(self._env_name, status="error", image_tag=None, container_id=None, container_port=None)


async def _off_loop(fn):
    return await asyncio.get_running_loop().run_in_executor(None, fn)


async def _build_and_run(env_name: str, context_dir: Path) -> tuple[str, str, int]:
    """Build the image off the event loop and start its container."""
    from forge.envgen.container import ContainerRuntime

    runtime = ContainerRuntime()

    def docker_ops() -> tuple[str, str, int]:
        image_tag = runtime.build(env_name, context_dir)
        container_id, port = runtime.run(env_name, image_tag)
        return image_tag, container_id, port

    return await _off_loop(docker_ops)


async def build_premade(request: BuildRequest, reporter: BuildReporter) -> None:
    template = request.env_type[len("premade:"):]
    reporter.log(f"[forge] setting up premade '{template}' environment…")
    reporter.building()
    premade_dir = Path(__file__).parent.parent.parent.parent / "docker" / "premade" / template
    if not premade_dir.exists():
        raise FileNotFoundError(f"Premade template '{template}' not found at {premade_dir}")
    reporter.log(f"[forge] building Docker image for '{template}'…")
    image_tag, container_id, port = await _build_and_run(request.env_name, premade_dir)
    reporter.running(container_id, port, image_tag)
    reporter.log(f"[forge] {template} environment ready on port {port} ✓")


async def build_cli(request: BuildRequest, reporter: BuildReporter) -> None:
    from forge.envgen.container import ContainerRuntime

    reporter.log(f"[forge] setting up CLI environment '{request.env_name}'…")
    reporter.building()
    reporter.log("[forge] pulling ubuntu:22.04 (first run may take a moment)…")
    container_id, port = await _off_loop(lambda: ContainerRuntime().run_cli(request.env_name))
    reporter.running(container_id, port, "builtin:cli")
    reporter.log("[forge] CLI container ready ✓")


async def build_browser(request: BuildRequest, reporter: BuildReporter) -> None:
    from forge.envgen.container import ContainerRuntime

    reporter.log(f"[forge] setting up Browser environment '{request.env_name}'…")
    reporter.building()
    reporter.log("[forge] pulling linuxserver/chromium (first run may take several minutes)…")
    container_id, port = await _off_loop(lambda: ContainerRuntime().run_browser(request.env_name))
    reporter.running(container_id, port, "builtin:browser")
    reporter.log(f"[forge] Browser container ready on port {port} ✓")


async def build_generated(request: BuildRequest, reporter: BuildReporter) -> None:
    """Extract, run the agent pipeline, build the app, then validate it live."""
    from backend.app.services import extraction_service
    from backend.app.services.env_orchestrator import EnvironmentOrchestrator
    from forge.settings import determinism_enabled

    env_name = request.env_name
    logger.info("[task:build_sandbox] _orchestrate started for %s", env_name)
    reporter.log(f"[forge] picked up job for '{env_name}'")
    reporter.building()

    async def on_progress(artifact_name: str, value) -> None:
        if artifact_name == "generation_plan":
            reporter.publish(pipeline_flags(value))
        label = _ARTIFACT_LABELS.get(artifact_name, artifact_name)
        reporter.log(f"[agent] {label} — done ✓")
        reporter.publish({"artifact": artifact_name, "status": "done"})

    async def on_agent_log(message: str) -> None:
        reporter.log(message)

    reporter.log("[forge] running extraction (LLM pass 1)…")
    compiler_input = await _off_loop(lambda: extraction_service.run_extraction(
        prompt=request.description,
        project_name=env_name,
        domain=request.domain or "localhost",
    ))
    reporter.log("[forge] extraction complete — starting agents…")

    await EnvironmentOrchestrator(on_progress=on_progress, on_log=on_agent_log).run(
        env_name=env_name,
        description=request.description,
        compiler_input=compiler_input,
        policy_requirements=request.policy_requirements,
        reward_requirements=request.reward_requirements,
        reference_urls=list(request.reference_urls),
        use_user_researcher=request.use_user_researcher,
        with_ui=request.with_ui,
        source_product_name=request.source_product_name,
        source_product_url=request.source_product_url,
    )
    reporter.log("[forge] all agents finished — building Docker image…")

    env_dir = generated_envs_root() / env_name
    if request.personas:
        written = _write_personas(env_dir, request.personas)
        if written:
            reporter.log(f"[forge] wrote {written} simulated {'person' if written == 1 else 'people'} — pick what they can do on the Simulated People page ✓")
    image_tag, container_id, port = await _build_and_run(env_name, env_dir / "app")
    reporter.log(f"[forge] container running on port {port} ✓")
    reporter.running(container_id, port, image_tag)

    base_url = f"http://localhost:{port}"
    if determinism_enabled():
        await _check_correctness(base_url, [a.name for a in compiler_input.actions], reporter.publish)
    else:
        reporter.log("[forge] determinism correctness gate disabled for experiment")

    await _validate_manifest(
        env_name=env_name,
        description=request.description,
        compiler_input=compiler_input,
        base_url=base_url,
        env_dir=env_dir,
        publish=reporter.publish,
    )


def _builder_for(env_type: str):
    if env_type.startswith("premade:"):
        return build_premade
    return {"cli": build_cli, "browser": build_browser}.get(env_type, build_generated)


@celery.task(name="backend.app.worker.tasks.build_sandbox_task", ignore_result=True)
def build_sandbox_task(
    job_id: str,
    env_name: str,
    env_type: str = "general",
    description: str = "",
    domain: str = "localhost",
    policy_requirements: str = "",
    reward_requirements: str = "",
    reference_urls: list[str] | None = None,
    use_user_researcher: bool = False,
    with_ui: bool = False,
    source_product_name: str = "",
    source_product_url: str = "",
    personas: dict | None = None,
) -> None:
    """Run sandbox build in a worker, stream progress via Redis pub/sub.

    env_type controls which build path runs:
      "premade:<name>" — build a bundled replica app
      "cli"     — pull ubuntu:22.04 and start a shell container
      "browser" — pull linuxserver/chromium and start a VNC browser container
      "general" — full LLM orchestration + Docker build (original flow)
    """
    import redis as _redis

    logger.info("[task:build_sandbox] STARTED — env_name=%s env_type=%s job_id=%s", env_name, env_type, job_id)
    request = BuildRequest(
        env_name=env_name,
        env_type=env_type,
        description=description,
        domain=domain,
        policy_requirements=policy_requirements,
        reward_requirements=reward_requirements,
        reference_urls=tuple(reference_urls or ()),
        use_user_researcher=use_user_researcher,
        with_ui=with_ui,
        source_product_name=source_product_name,
        source_product_url=source_product_url,
        personas=personas,
    )
    client = _redis.from_url(redis_url())
    channel = f"forge:progress:{env_name}"
    reporter = BuildReporter(env_name, lambda message: client.publish(channel, json.dumps(message)))
    try:
        asyncio.run(_builder_for(env_type)(request, reporter))
        logger.info("[task:build_sandbox] COMPLETED — env_name=%s", env_name)
        reporter.publish({"done": True})
    except Exception as exc:
        logger.exception("[task:build_sandbox] FAILED — env_name=%s error=%s", env_name, exc)
        reporter.log(f"[forge] ERROR: {exc}")
        reporter.failed()
        reporter.publish({"done": True, "error": f"Build failed: {exc}"})
    finally:
        client.close()


@celery.task(name="backend.app.worker.tasks.cleanup_expired_sandboxes")
def cleanup_expired_sandboxes() -> None:
    from datetime import datetime, timezone
    from backend.app.models import SandboxEnvironment
    from forge.envgen.container import ContainerRuntime
    from backend.app.database import get_session_factory

    SessionLocal = get_session_factory()
    with SessionLocal() as db:
        expired = (
            db.query(SandboxEnvironment)
            .filter(
                SandboxEnvironment.expires_at <= datetime.now(timezone.utc),
                SandboxEnvironment.status.notin_(["expired", "deleted"]),
            )
            .all()
        )
        if not expired:
            return
        runtime = ContainerRuntime()
        for sandbox in expired:
            if sandbox.container_id:
                runtime.remove(sandbox.container_id, sandbox.image_tag)
            sandbox.status = "expired"
        db.commit()

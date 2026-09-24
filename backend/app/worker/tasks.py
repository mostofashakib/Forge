from __future__ import annotations
import asyncio
import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path

from celery import group
from sqlalchemy import update

from backend.app.worker.celery_app import celery
from backend.app.models import RolloutJob, Episode, AgentRun, AgentEpisode
from backend.app.utils.env_loader import load_forge_env
from forge.settings import generated_envs_root, redis_url

logger = logging.getLogger(__name__)


@celery.task(bind=True)
def run_episode_task(self, rollout_job_id: str, episode_index: int, seed: int) -> str:
    """Run a single episode for a RolloutJob. Returns episode_id."""
    from backend.app.services.episode_collector import EpisodeDataCollector
    from forge.runtime.agents.factory import make_agent
    from forge.runtime.agent_logger import AgentRunLogger, run_logged_episode
    from backend.app.database import get_session_factory

    SessionLocal = get_session_factory()

    episode_id = f"ep_{seed:08x}_{secrets.token_hex(4)}"

    with SessionLocal() as db:
        job = db.get(RolloutJob, rollout_job_id)
        if job is None:
            logger.error("RolloutJob %s not found", rollout_job_id)
            return episode_id
        env_name = job.env_name
        task_name = job.task_name
        agent_id = job.agent_id

    envs_root = generated_envs_root()
    jsonl_dir = envs_root / env_name / "episodes"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / f"{episode_id}.jsonl"

    with SessionLocal() as db:
        db.add(Episode(
            id=episode_id,
            env_name=env_name,
            task_name=task_name,
            seed=seed,
            agent_id=agent_id,
            status="running",
            total_steps=0,
            total_reward=0.0,
            passed=False,
            started_at=datetime.now(timezone.utc),
            jsonl_path=str(jsonl_path),
        ))
        db.commit()

    with SessionLocal() as db_ep:
        try:
            telemetry = EpisodeDataCollector(
                episode_id=episode_id,
                db_session=db_ep,
                jsonl_path=jsonl_path,
            )
            env = load_forge_env(env_name, telemetry)
            try:
                selected_task = env.task_source.get(task_name)
            except KeyError:
                # Older generated packages may expose only a default task. Keep
                # the requested rollout objective aligned for both reset and
                # prompting even when that package predates TaskSource.
                available_tasks = env.task_source.tasks()
                selected_task = available_tasks[0] if available_tasks else {
                    "id": task_name,
                    "objective": task_name,
                }
            agent = make_agent(
                agent_id,
                environment=env,
                task=selected_task,
            )

            # Drive the episode through the run logger so the full trace (LLM
            # calls, actions, and state changes) is captured. The trace is
            # persisted in `finally` so an aborted run still leaves a partial one.
            run_logger = AgentRunLogger(run_id=episode_id)
            trace_path = jsonl_path.with_name(f"{episode_id}.trace.jsonl")
            try:
                run_logged_episode(
                    env,
                    agent,
                    run_logger,
                    seed=seed,
                    task=selected_task,
                )
            finally:
                trace_path.write_text(run_logger.to_jsonl())
            # ForgeEnv.step() calls telemetry.complete_episode() on termination

        except Exception as exc:
            logger.exception("Episode %s failed: %s", episode_id, exc)
            ep = db_ep.get(Episode, episode_id)
            if ep is not None:
                ep.status = "failed"
                ep.completed_at = datetime.now(timezone.utc)
                db_ep.commit()

    # Atomically increment episodes_completed; mark job completed when all done
    with SessionLocal() as db2:
        db2.execute(
            update(RolloutJob)
            .where(RolloutJob.id == rollout_job_id)
            .values(episodes_completed=RolloutJob.episodes_completed + 1)
        )
        db2.commit()
        job2 = db2.get(RolloutJob, rollout_job_id)
        if job2 and job2.episodes_completed >= job2.num_episodes:
            job2.status = "completed"
            job2.completed_at = datetime.now(timezone.utc)
            db2.commit()

    return episode_id


def _load_manifest(env_dir: Path):
    """The env's state schema manifest, or None when it has none or it is unreadable."""
    from forge.schema.state_schema import StateSchemaManifest

    manifest_path = env_dir / "state_schema.json"
    if not manifest_path.exists():
        return None
    try:
        return StateSchemaManifest.model_validate_json(manifest_path.read_text())
    except Exception as exc:
        logger.warning("[manifest] could not load %s: %s", manifest_path, exc)
        return None


def _load_personas(env_dir):
    """The cast configured for this environment, or None if it has none.

    Read here rather than inside the runner so a malformed `personas:` block
    degrades an agent run to an empty environment with a logged warning,
    instead of failing every episode of the run.
    """
    config_path = env_dir / "custom" / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml

        from forge.personas.config import load_population

        raw = yaml.safe_load(config_path.read_text()) or {}
        population = load_population(raw.get("personas"))
    except Exception as exc:
        logger.warning("[personas] could not load cast for %s: %s", env_dir.name, exc)
        return None
    return population if population.enabled else None


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

    manifest = _load_manifest(env_dir)
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
      "cli"     — pull ubuntu:22.04 and start a shell container
      "browser" — pull linuxserver/chromium and start a VNC browser container
      "general" — full LLM orchestration + Docker build (original flow)
    """
    import redis as _redis
    from forge.envgen.container import ContainerRuntime

    logger.info("[task:build_sandbox] STARTED — env_name=%s env_type=%s job_id=%s", env_name, env_type, job_id)

    redis_connection_url = redis_url()
    r = _redis.from_url(redis_connection_url)
    channel = f"forge:progress:{env_name}"

    def publish(msg: dict) -> None:
        r.publish(channel, json.dumps(msg))

    def _set_status(status: str) -> None:
        _update_sandbox(env_name, status=status)

    def _set_running(container_id: str, port: int, image_tag: str) -> None:
        _update_sandbox(
            env_name,
            status="running",
            container_id=container_id,
            container_port=port or None,
            image_tag=image_tag,
        )

    async def _build_premade() -> None:
        template = env_type[len("premade:"):]
        publish({"log": f"[forge] setting up premade '{template}' environment…"})
        _set_status("building")
        project_root = Path(__file__).parent.parent.parent.parent
        premade_dir = project_root / "docker" / "premade" / template
        if not premade_dir.exists():
            raise FileNotFoundError(f"Premade template '{template}' not found at {premade_dir}")
        publish({"log": f"[forge] building Docker image for '{template}'…"})
        runtime = ContainerRuntime()
        loop = asyncio.get_running_loop()

        def _docker_ops() -> tuple[str, str, int]:
            image_tag = runtime.build(env_name, premade_dir)
            container_id, port = runtime.run(env_name, image_tag)
            return image_tag, container_id, port

        image_tag, container_id, port = await loop.run_in_executor(None, _docker_ops)
        _set_running(container_id, port, image_tag)
        publish({"log": f"[forge] {template} environment ready on port {port} ✓"})

    async def _build_cli() -> None:
        publish({"log": f"[forge] setting up CLI environment '{env_name}'…"})
        _set_status("building")
        publish({"log": "[forge] pulling ubuntu:22.04 (first run may take a moment)…"})
        loop = asyncio.get_running_loop()
        container_id, port = await loop.run_in_executor(None, lambda: ContainerRuntime().run_cli(env_name))
        _set_running(container_id, port, "builtin:cli")
        publish({"log": "[forge] CLI container ready ✓"})

    async def _build_browser() -> None:
        publish({"log": f"[forge] setting up Browser environment '{env_name}'…"})
        _set_status("building")
        publish({"log": "[forge] pulling linuxserver/chromium (first run may take several minutes)…"})
        loop = asyncio.get_running_loop()
        container_id, port = await loop.run_in_executor(None, lambda: ContainerRuntime().run_browser(env_name))
        _set_running(container_id, port, "builtin:browser")
        publish({"log": f"[forge] Browser container ready on port {port} ✓"})

    async def _orchestrate() -> None:
        from backend.app.services import extraction_service
        from backend.app.services.env_orchestrator import EnvironmentOrchestrator

        logger.info("[task:build_sandbox] _orchestrate started for %s", env_name)
        publish({"log": f"[forge] picked up job for '{env_name}'"})
        _set_status("building")

        async def on_progress(artifact_name: str, _value) -> None:
            if artifact_name == "generation_plan":
                publish(pipeline_flags(_value))
            label = _ARTIFACT_LABELS.get(artifact_name, artifact_name)
            publish({"log": f"[agent] {label} — done ✓"})
            publish({"artifact": artifact_name, "status": "done"})

        async def on_agent_log(message: str) -> None:
            publish({"log": message})

        publish({"log": "[forge] running extraction (LLM pass 1)…"})
        loop = asyncio.get_running_loop()
        compiler_input = await loop.run_in_executor(
            None,
            lambda: extraction_service.run_extraction(
                prompt=description,
                project_name=env_name,
                domain=domain or "localhost",
            ),
        )
        publish({"log": "[forge] extraction complete — starting agents…"})

        orchestrator = EnvironmentOrchestrator(on_progress=on_progress, on_log=on_agent_log)
        await orchestrator.run(
            env_name=env_name,
            description=description,
            compiler_input=compiler_input,
            policy_requirements=policy_requirements,
            reward_requirements=reward_requirements,
            reference_urls=reference_urls or [],
            use_user_researcher=use_user_researcher,
            with_ui=with_ui,
            source_product_name=source_product_name,
            source_product_url=source_product_url,
        )
        publish({"log": "[forge] all agents finished — building Docker image…"})

        envs_root = generated_envs_root()
        if personas:
            written = _write_personas(envs_root / env_name, personas)
            if written:
                publish({"log": f"[forge] wrote {written} simulated {'person' if written == 1 else 'people'} — pick what they can do on the Simulated People page ✓"})
        app_dir = envs_root / env_name / "app"
        runtime = ContainerRuntime()

        def _docker_ops() -> tuple[str, str, int]:
            image_tag = runtime.build(env_name, app_dir)
            container_id, port = runtime.run(env_name, image_tag)
            return image_tag, container_id, port

        image_tag, container_id, port = await loop.run_in_executor(None, _docker_ops)
        publish({"log": f"[forge] container running on port {port} ✓"})
        _set_running(container_id, port, image_tag)

        from forge.settings import determinism_enabled
        base_url = f"http://localhost:{port}"
        if determinism_enabled():
            await _check_correctness(base_url, [a.name for a in compiler_input.actions], publish)
        else:
            publish({"log": "[forge] determinism correctness gate disabled for experiment"})

        await _validate_manifest(
            env_name=env_name,
            description=description,
            compiler_input=compiler_input,
            base_url=base_url,
            env_dir=envs_root / env_name,
            publish=publish,
        )

    if env_type.startswith("premade:"):
        _build_fn = _build_premade
    else:
        _build_fn = {"cli": _build_cli, "browser": _build_browser}.get(env_type, _orchestrate)

    try:
        asyncio.run(_build_fn())
        logger.info("[task:build_sandbox] COMPLETED — env_name=%s", env_name)
        publish({"done": True})
    except Exception as exc:
        logger.exception("[task:build_sandbox] FAILED — env_name=%s error=%s", env_name, exc)
        publish({"log": f"[forge] ERROR: {exc}"})
        # Clear container/image references too — otherwise a leftover tag from
        # a previous successful build stays in the DB, and /start would later
        # try to spin up a container against an image that may no longer exist.
        _update_sandbox(env_name, status="error", image_tag=None, container_id=None, container_port=None)
        publish({"done": True, "error": f"Build failed: {exc}"})
    finally:
        r.close()


@celery.task
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


@celery.task(bind=True)
def run_container_episode_task(self, run_id: str, episode_index: int, seed: int) -> str:
    """Run a single agent episode against a containerized environment.

    Routes to the appropriate runner based on env_type:
      "general" → ContainerEpisodeRunner (HTTP FastAPI)
      "cli"     → CliEpisodeRunner (docker exec shell)
      "browser" → BrowserEpisodeRunner (Playwright CDP)
    """
    from backend.app.database import get_session_factory
    from backend.app.models import SandboxEnvironment
    from forge.settings import experiment_seed

    SessionLocal = get_session_factory()
    episode_id = f"cep_{seed:08x}_{secrets.token_hex(4)}"

    with SessionLocal() as db:
        run = db.get(AgentRun, run_id)
        if run is None:
            logger.error("[container-ep] AgentRun %s not found", run_id)
            return episode_id
        env_name = run.env_name
        sb = db.get(SandboxEnvironment, env_name)
        if sb is None or sb.container_id is None:
            logger.error("[container-ep] sandbox %s has no running container", env_name)
            now = datetime.now(timezone.utc)
            db.add(AgentEpisode(
                id=episode_id,
                run_id=run_id,
                episode_index=episode_index,
                seed=seed,
                status="failed",
                termination_reason=f"sandbox {env_name} has no running container",
                started_at=now,
                completed_at=now,
            ))
            db.commit()
            _count_finished_episode(run_id)
            return episode_id
        env_type = sb.env_type
        container_id = sb.container_id
        container_port = sb.container_port
        agent_id = run.agent_id
        objective = run.objective
        max_steps = run.max_steps
        divergence_threshold = run.divergence_threshold
        consecutive_below_threshold = run.consecutive_below_threshold
        dead_end_patience = run.dead_end_patience
        success_threshold = run.success_threshold

    envs_root = generated_envs_root()
    jsonl_dir = envs_root / env_name / "agent_episodes"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / f"{episode_id}.jsonl"

    with SessionLocal() as db:
        db.add(AgentEpisode(
            id=episode_id,
            run_id=run_id,
            episode_index=episode_index,
            seed=seed,
            status="running",
            started_at=datetime.now(timezone.utc),
            jsonl_path=str(jsonl_path),
        ))
        db.commit()

    try:
        if env_type == "cli":
            from forge.envgen.cli_runner import CliEpisodeRunner, CliEpisodeConfig
            from forge.envgen.agents.cli_agent import make_cli_agent, ReplayCliAgent
            from forge.envgen.tiered_reward import TieredRewardEngine, TieredRewardConfig
            cfg = CliEpisodeConfig(
                container_id=container_id,
                objective=objective,
                max_steps=max_steps,
                divergence_threshold=divergence_threshold,
                consecutive_below_threshold=consecutive_below_threshold,
                dead_end_patience=dead_end_patience,
                success_threshold=success_threshold,
            )
            import json as _json
            from backend.app.services.reward_config import load_reward_config
            reward_cfg = load_reward_config(env_name)
            reward_engine = TieredRewardEngine(
                config=TieredRewardConfig.from_preset(
                    reward_cfg.reward_preset,
                    partial_credit_methods=reward_cfg.scoring_methods,
                )
            )
            replay_path = envs_root / env_name / "synthetic_replay.json"
            if replay_path.exists():
                manifest = _json.loads(replay_path.read_text(encoding="utf-8"))
                trajectory_episodes = manifest.get("episodes", [])
                if trajectory_episodes:
                    ep_commands = trajectory_episodes[seed % len(trajectory_episodes)]
                    agent = ReplayCliAgent(ep_commands)
                    logger.info(
                        "[container-ep] using replay agent seed=%d → trajectory %d (%d commands)",
                        seed, seed % len(trajectory_episodes), len(ep_commands),
                    )
                else:
                    agent = make_cli_agent(agent_id, seed=experiment_seed(seed))
            else:
                agent = make_cli_agent(agent_id, seed=experiment_seed(seed))
            result = CliEpisodeRunner(cfg, reward_engine=reward_engine).run_episode(
                agent, episode_id=episode_id, jsonl_path=jsonl_path
            )

        elif env_type == "browser":
            import docker as _docker
            dc = _docker.from_env()
            try:
                c = dc.containers.get(container_id)
                c.reload()
                cdp_mapping = c.ports.get("9222/tcp")
            finally:
                dc.close()
            if not cdp_mapping:
                raise RuntimeError(
                    "CDP port 9222 is not mapped on the browser container. "
                    "Recreate the environment to pick up the new CDP configuration."
                )
            cdp_port = int(cdp_mapping[0]["HostPort"])
            from forge.envgen.browser_runner import BrowserEpisodeRunner, BrowserEpisodeConfig
            from forge.envgen.agents.browser_agent import make_browser_agent
            cfg = BrowserEpisodeConfig(
                cdp_url=f"http://localhost:{cdp_port}",
                objective=objective,
                max_steps=max_steps,
                divergence_threshold=divergence_threshold,
                consecutive_below_threshold=consecutive_below_threshold,
                dead_end_patience=dead_end_patience,
                success_threshold=success_threshold,
            )
            agent = make_browser_agent(agent_id, seed=experiment_seed(seed))
            result = BrowserEpisodeRunner(cfg).run_episode(
                agent, episode_id=episode_id, jsonl_path=jsonl_path
            )

        else:  # general / premade (both run FastAPI over HTTP)
            from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig
            from forge.envgen.agents.container_agent import make_container_agent
            if container_port is None:
                raise RuntimeError(f"General sandbox {env_name} has no container_port")
            cfg = EpisodeConfig(
                base_url=f"http://localhost:{container_port}",
                objective=objective,
                max_steps=max_steps,
                divergence_threshold=divergence_threshold,
                consecutive_below_threshold=consecutive_below_threshold,
                dead_end_patience=dead_end_patience,
                success_threshold=success_threshold,
                personas=_load_personas(envs_root / env_name),
            )
            # The manifest enables HashNormalizer + StateDiffFloor when present.
            manifest = _load_manifest(envs_root / env_name)
            agent = make_container_agent(agent_id, seed=experiment_seed(seed))
            with ContainerEpisodeRunner(cfg, manifest=manifest) as runner:
                result = runner.run_episode(agent, episode_id=episode_id, jsonl_path=jsonl_path)

        with SessionLocal() as db:
            ep = db.get(AgentEpisode, episode_id)
            if ep is not None:
                ep.status = "completed"
                ep.total_steps = len(result.steps)
                ep.total_reward = result.total_reward
                ep.final_objective_score = result.final_objective_score
                ep.termination_reason = result.termination_reason
                ep.completed_at = datetime.now(timezone.utc)
                db.commit()

    except Exception as exc:
        logger.exception("[container-ep] episode %s failed: %s", episode_id, exc)
        with SessionLocal() as db:
            ep = db.get(AgentEpisode, episode_id)
            if ep is not None:
                ep.status = "failed"
                ep.termination_reason = str(exc)[:255]
                ep.completed_at = datetime.now(timezone.utc)
                db.commit()

    _count_finished_episode(run_id)
    return episode_id


def _count_finished_episode(run_id: str) -> None:
    """Atomically increment the run counter; mark the run completed when all are done."""
    from backend.app.database import get_session_factory

    with get_session_factory()() as db:
        db.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(episodes_completed=AgentRun.episodes_completed + 1)
        )
        db.commit()
        run = db.get(AgentRun, run_id)
        if run and run.episodes_completed >= run.num_episodes:
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            db.commit()


@celery.task(bind=True)
def run_container_run_task(self, run_id: str) -> None:
    """Dispatch all episode subtasks for an AgentRun."""
    from backend.app.database import get_session_factory

    logger.info("[container-run] STARTED — run_id=%s", run_id)
    SessionLocal = get_session_factory()

    with SessionLocal() as db:
        run = db.get(AgentRun, run_id)
        if run is None:
            logger.error("[container-run] AgentRun %s not found", run_id)
            return
        run.status = "running"
        num_episodes = run.num_episodes
        seed_start = run.seed_start
        db.commit()

    try:
        subtasks = group(
            run_container_episode_task.s(run_id, i, seed_start + i)
            for i in range(num_episodes)
        )
        subtasks.apply_async()
    except Exception as exc:
        logger.exception("[container-run] dispatch failed for %s: %s", run_id, exc)
        with SessionLocal() as db:
            run_fail = db.get(AgentRun, run_id)
            if run_fail is not None:
                run_fail.status = "failed"
                run_fail.error = str(exc)
                run_fail.completed_at = datetime.now(timezone.utc)
                db.commit()


@celery.task(bind=True)
def run_rollout_task(self, rollout_job_id: str) -> None:
    """Dispatch all episode subtasks for a RolloutJob."""
    from backend.app.database import get_session_factory

    logger.info("[task:run_rollout] STARTED — rollout_job_id=%s", rollout_job_id)
    SessionLocal = get_session_factory()

    with SessionLocal() as db:
        job = db.get(RolloutJob, rollout_job_id)
        if job is None:
            logger.error("[task:run_rollout] RolloutJob %s not found", rollout_job_id)
            return
        job.status = "running"
        num_episodes = job.num_episodes
        seed_start = job.seed_start
        db.commit()
        logger.info("[task:run_rollout] dispatching %d episodes for job %s", num_episodes, rollout_job_id)

    try:
        subtasks = group(
            run_episode_task.s(rollout_job_id, i, seed_start + i)
            for i in range(num_episodes)
        )
        subtasks.apply_async()
    except Exception as exc:
        logger.exception("RolloutJob %s dispatch failed: %s", rollout_job_id, exc)
        with SessionLocal() as db_fail:
            job_fail = db_fail.get(RolloutJob, rollout_job_id)
            if job_fail is not None:
                job_fail.status = "failed"
                job_fail.error = str(exc)
                job_fail.completed_at = datetime.now(timezone.utc)
                db_fail.commit()


@celery.task(bind=True)
def run_benchmark_task(
    self,
    run_id: str,
    domains: list[str],
    depth: int,
    seeds: int,
    output_dir: str,
) -> None:
    """Collect benchmark episodes and compute env quality metrics.

    Streams progress to Redis pub/sub channel forge:benchmark:{run_id}.
    Message types:
      {"total": N}           — total episodes to run
      {"log": "..."}         — human-readable log line
      {"progress": K}        — K episodes completed so far
      {"done": true}         — run complete
      {"error": "..."}       — run failed
    """
    import json as _json
    import redis as _redis
    from functools import cache
    from pathlib import Path as _Path

    redis_connection_url = redis_url()
    channel = f"forge:benchmark:{run_id}"

    try:
        r = _redis.from_url(redis_connection_url, socket_connect_timeout=3, socket_timeout=3)
        r.ping()
    except Exception as exc:
        logger.error("[task:benchmark] Redis unavailable — %s", exc)
        _update_run_status(run_id, "failed", error=str(exc))
        return

    def publish(msg: dict) -> None:
        try:
            r.publish(channel, _json.dumps(msg))
        except Exception:
            logger.debug("[task:benchmark] progress publish failed", exc_info=True)

    _update_run_status(run_id, "running")
    publish({"log": f"[benchmark] starting run {run_id} — domains={domains} depth={depth} seeds={seeds}"})

    try:
        from forge.benchmark.data_collector import DataCollector, CollectionConfig, CollectionCheckpoint
        from forge.benchmark.env_quality import compute_env_quality
        from forge.benchmark.report import BenchmarkReport, ReportConfig

        from backend.app.database import get_session_factory
        from forge.benchmark.compiled_tasks import (
            CompiledTaskProvider,
            db_compiler_input_loader,
        )

        output_path = _Path(output_dir)
        cfg = CollectionConfig(domains=domains, depth=depth, seeds=seeds, output_dir=output_path / "data")
        # Each selected environment is benchmarked against its own compiled tasks.
        task_provider = CompiledTaskProvider(
            loader=db_compiler_input_loader(get_session_factory())
        )
        collector = DataCollector(cfg, task_provider=task_provider)
        envs_root = generated_envs_root()

        @cache
        def manifest_for(domain: str):
            return _load_manifest(envs_root / domain)

        for domain in domains:
            if not task_provider.tasks_for(domain=domain, depth=depth):
                publish({"log": f"  [skip] '{domain}' has no tasks at depth {depth} — compile/select an environment with tasks"})

        checkpoint = CollectionCheckpoint(output_dir=output_path / "data")
        pending = collector.pending_runs(checkpoint)
        total = len(pending)
        publish({"total": total})
        publish({"log": f"[benchmark] {total} episodes pending"})

        completed = 0

        def run_episode(task, seed, jsonl_path):
            nonlocal completed
            from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig
            from forge.envgen.agents.container_agent import make_container_agent
            from forge.settings import experiment_seed

            manifest = manifest_for(task.domain)
            # Read per episode: a restarted environment comes back on a new port.
            port_file = envs_root / task.domain / "port"
            if not port_file.exists():
                publish({"log": f"  [skip] no port file for domain '{task.domain}' — start that environment first"})
                return

            port = int(port_file.read_text().strip())
            cfg_ep = EpisodeConfig(base_url=f"http://localhost:{port}", objective=task.objective)
            agent = make_container_agent("random", seed=experiment_seed(seed))
            with ContainerEpisodeRunner(cfg_ep, manifest=manifest) as runner:
                result = runner.run_episode(agent, jsonl_path=jsonl_path, seed=seed)

            completed += 1
            publish({"log": f"  {task.name} seed={seed}  reward={result.total_reward:.3f}  reason={result.termination_reason}"})
            publish({"progress": completed})

        collector.collect(run_episode)

        metrics = []
        for domain in domains:
            manifest = manifest_for(domain)
            if manifest:
                m = compute_env_quality(episode_dir=output_path / "data" / domain, manifest=manifest)
                metrics.append(m)
                publish({"log": f"  {domain}: coverage={m.state_coverage_score:.2f}  dead_end_rate={m.dead_end_rate:.2f}"})

        report = BenchmarkReport(ReportConfig(output_dir=output_path))
        report.write_env_quality(metrics)

        report_data = [
            {
                "env_name": m.env_name,
                "state_coverage_score": m.state_coverage_score,
                "reward_density": m.reward_density,
                "dead_end_rate": m.dead_end_rate,
                "action_diversity": m.action_diversity,
                "num_episodes": m.num_episodes,
                "num_steps": m.num_steps,
            }
            for m in metrics
        ]
        _update_run_status(run_id, "done", report_json=_json.dumps(report_data))
        publish({"done": True, "log": f"[benchmark] run complete — {len(metrics)} environments analyzed"})
        logger.info("[task:benchmark] run %s complete", run_id)

    except Exception as exc:
        logger.exception("[task:benchmark] run %s failed: %s", run_id, exc)
        _update_run_status(run_id, "failed", error=str(exc))
        publish({"error": str(exc)})


def _harbor_command(config: dict, executable: str) -> tuple[list[str], Path]:
    """Build a Harbor invocation without passing user input through a shell."""
    task_path = Path(config["harbor_task_path"]).resolve()
    agent = str(config["harbor_agent"])
    command = [executable, "run", "-p", task_path.name]
    if ":" in agent:
        command.extend(["--agent-import-path", agent])
    else:
        command.extend(["--agent", agent])
    command.extend(["--model", str(config["harbor_model"])])
    return command, task_path.parent


@celery.task(name="backend.app.worker.tasks.run_evaluation_task", ignore_result=True)
def run_evaluation_task(run_id: str, engine: str, config: dict) -> None:
    """Run a held-out Forge eval or an optional Harbor job with streamed logs."""
    import json as _json
    import redis as _redis

    channel = f"forge:benchmark:{run_id}"
    try:
        redis_client = _redis.from_url(
            redis_url(), socket_connect_timeout=3, socket_timeout=3
        )
        redis_client.ping()
    except Exception as exc:
        logger.error("[task:eval] Redis unavailable — %s", exc)
        _update_run_status(run_id, "failed", error=str(exc))
        return

    def publish(message: dict) -> None:
        try:
            redis_client.publish(channel, _json.dumps(message))
        except Exception:
            logger.debug("[task:eval] progress publish failed", exc_info=True)

    _update_run_status(run_id, "running")
    publish({"log": f"[eval] starting {engine} evaluation {run_id}"})
    try:
        if engine == "forge":
            from forge.benchmark._eval import evaluate_on_suite

            result = evaluate_on_suite(
                model_path=config["checkpoint"],
                suite=config["experiment"],
                seed=config.get("seed"),
                runs_dir=config["runs_dir"],
                run_id=run_id,
            )
            publish({
                "log": (
                    "[eval] held-out pass rate "
                    f"{result['heldout_pass_rate']:.3f}"
                )
            })
        elif engine == "harbor":
            import shutil
            import subprocess

            bundled = Path.cwd() / "example_tasks" / ".venv" / "bin" / "harbor"
            executable = shutil.which("harbor")
            if executable is None and bundled.is_file():
                executable = str(bundled)
            if executable is None:
                raise RuntimeError(
                    "Harbor is not installed. Run `uv tool install harbor` first."
                )
            command, working_dir = _harbor_command(config, executable)
            publish({"log": f"[eval] Harbor task: {working_dir / command[3]}"})
            process = subprocess.Popen(
                command,
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is not None:
                for line in process.stdout:
                    publish({"log": line.rstrip()})
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"Harbor exited with status {return_code}")
            result = {
                "engine": "harbor",
                "status": "completed",
                "result_path": str(working_dir / "jobs"),
                "task_path": config["harbor_task_path"],
                "agent": config["harbor_agent"],
                "model": config["harbor_model"],
            }
        else:
            raise ValueError(f"unsupported evaluation engine: {engine}")

        _update_run_status(run_id, "done", report_json=_json.dumps(result))
        publish({"done": True, "result": result, "log": "[eval] evaluation complete"})
    except Exception as exc:
        logger.exception("[task:eval] run %s failed: %s", run_id, exc)
        _update_run_status(run_id, "failed", error=str(exc))
        publish({"error": str(exc)})


def _update_run_status(
    run_id: str,
    status: str,
    error: str | None = None,
    report_json: str | None = None,
) -> None:
    from backend.app.database import get_session_factory
    from backend.app.models import BenchmarkRun
    from datetime import datetime, timezone

    try:
        SessionLocal = get_session_factory()
        with SessionLocal() as db:
            run = db.get(BenchmarkRun, run_id)
            if run:
                run.status = status
                if error is not None:
                    run.error = error
                if report_json is not None:
                    run.report_json = report_json
                if status in ("done", "failed"):
                    run.completed_at = datetime.now(timezone.utc)
                db.commit()
    except Exception as exc:
        logger.error("[task:benchmark] DB update failed for %s: %s", run_id, exc)

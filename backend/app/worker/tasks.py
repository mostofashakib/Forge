from __future__ import annotations
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from celery import group
from sqlalchemy import update
from sqlalchemy.orm import Session

from backend.app.worker.celery_app import celery
from backend.app.models import RolloutJob, Episode, AgentRun, AgentEpisode
from backend.app.utils.env_loader import load_forge_env

logger = logging.getLogger(__name__)


@celery.task(bind=True)
def run_episode_task(self, rollout_job_id: str, episode_index: int, seed: int) -> str:
    """Run a single episode for a RolloutJob. Returns episode_id."""
    from forge.runtime.telemetry import TelemetryClient
    from forge.runtime.agents.factory import make_agent
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

    envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
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
            telemetry = TelemetryClient(
                episode_id=episode_id,
                db_session=db_ep,
                jsonl_path=jsonl_path,
            )
            env = load_forge_env(env_name, telemetry)
            agent = make_agent(agent_id)

            obs, _ = env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.act(obs, env.action_types)
                obs, _, terminated, truncated, _ = env.step(action)
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


@celery.task(name="backend.app.worker.tasks.build_sandbox_task", ignore_result=True)
def build_sandbox_task(
    job_id: str,
    env_name: str,
    env_type: str = "general",
    description: str = "",
    domain: str = "localhost",
    policy_requirements: str = "",
    reward_requirements: str = "",
) -> None:
    """Run sandbox build in a worker, stream progress via Redis pub/sub.

    env_type controls which build path runs:
      "cli"     — pull ubuntu:22.04 and start a shell container
      "browser" — pull linuxserver/chromium and start a VNC browser container
      "general" — full LLM orchestration + Docker build (original flow)
    """
    import asyncio
    import json
    import redis as _redis
    from backend.app.database import get_session_factory
    from backend.app.models import SandboxEnvironment
    from forge.envgen.container import ContainerRuntime

    logger.info("[task:build_sandbox] STARTED — env_name=%s env_type=%s job_id=%s", env_name, env_type, job_id)

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    r = _redis.from_url(redis_url)
    channel = f"forge:progress:{env_name}"

    def publish(msg: dict) -> None:
        r.publish(channel, json.dumps(msg))

    SessionLocal = get_session_factory()

    def _set_status(status: str) -> None:
        with SessionLocal() as db:
            sb = db.get(SandboxEnvironment, env_name)
            if sb:
                sb.status = status
                db.commit()

    def _set_running(container_id: str, port: int, image_tag: str) -> None:
        with SessionLocal() as db:
            sb = db.get(SandboxEnvironment, env_name)
            if sb:
                sb.status = "running"
                sb.container_id = container_id
                sb.container_port = port or None
                sb.image_tag = image_tag
                db.commit()

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
            label = {
                "app_code":          "App Generator",
                "instrumented_code": "Telemetry Instrumentation",
                "state_bridge_code": "State Bridge",
                "policy_dsl":        "Policy Rules",
                "reward_fn_code":    "Reward Function",
            }.get(artifact_name, artifact_name)
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
        )
        publish({"log": "[forge] all agents finished — building Docker image…"})

        envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
        app_dir = envs_root / env_name / "app"
        runtime = ContainerRuntime()

        def _docker_ops() -> tuple[str, str, int]:
            image_tag = runtime.build(env_name, app_dir)
            container_id, port = runtime.run(env_name, image_tag)
            return image_tag, container_id, port

        image_tag, container_id, port = await loop.run_in_executor(None, _docker_ops)
        publish({"log": f"[forge] container running on port {port} ✓"})
        _set_running(container_id, port, image_tag)

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
        with SessionLocal() as db:
            sb = db.get(SandboxEnvironment, env_name)
            if sb:
                sb.status = "error"
                sb.image_tag = None
                sb.container_id = None
                sb.container_port = None
                db.commit()
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

    envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
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
            from forge.envgen.agents.cli_agent import make_cli_agent
            cfg = CliEpisodeConfig(
                container_id=container_id,
                objective=objective,
                max_steps=max_steps,
                divergence_threshold=divergence_threshold,
                consecutive_below_threshold=consecutive_below_threshold,
                dead_end_patience=dead_end_patience,
                success_threshold=success_threshold,
            )
            agent = make_cli_agent(agent_id, seed=seed)
            result = CliEpisodeRunner(cfg).run_episode(
                agent, episode_id=episode_id, jsonl_path=jsonl_path
            )

        elif env_type == "browser":
            import docker as _docker
            dc = _docker.from_env()
            c = dc.containers.get(container_id)
            c.reload()
            cdp_mapping = c.ports.get("9222/tcp")
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
            agent = make_browser_agent(agent_id, seed=seed)
            result = BrowserEpisodeRunner(cfg).run_episode(
                agent, episode_id=episode_id, jsonl_path=jsonl_path
            )

        else:  # general
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
            )
            agent = make_container_agent(agent_id, seed=seed)
            with ContainerEpisodeRunner(cfg) as runner:
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

    # Atomically increment run counter; mark run completed when all done
    with SessionLocal() as db:
        db.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(episodes_completed=AgentRun.episodes_completed + 1)
        )
        db.commit()
        run2 = db.get(AgentRun, run_id)
        if run2 and run2.episodes_completed >= run2.num_episodes:
            run2.status = "completed"
            run2.completed_at = datetime.now(timezone.utc)
            db.commit()

    return episode_id


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

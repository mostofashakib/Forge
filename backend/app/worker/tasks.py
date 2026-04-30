from __future__ import annotations
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from celery import group
from sqlalchemy import update

from backend.app.worker.celery_app import celery
from backend.app.models import RolloutJob, Episode
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


@celery.task(name="backend.app.worker.tasks.build_sandbox_task")
def build_sandbox_task(
    job_id: str,
    env_name: str,
    description: str,
    domain: str,
    policy_requirements: str,
    reward_requirements: str,
) -> None:
    """Run sandbox orchestration + Docker build in a worker, stream progress via Redis pub/sub."""
    import asyncio
    import json
    import redis as _redis
    from backend.app.database import get_session_factory
    from backend.app.models import SandboxEnvironment
    from backend.app.services import extraction_service
    from backend.app.services.env_orchestrator import EnvironmentOrchestrator
    from forge.envgen.container import ContainerRuntime

    logger.info("[task:build_sandbox] STARTED — env_name=%s job_id=%s", env_name, job_id)

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    r = _redis.from_url(redis_url)
    channel = f"forge:progress:{env_name}"

    def publish(msg: dict) -> None:
        r.publish(channel, json.dumps(msg))

    SessionLocal = get_session_factory()

    async def _orchestrate() -> None:
        logger.info("[task:build_sandbox] _orchestrate started for %s", env_name)
        publish({"log": f"[forge] picked up job for '{env_name}'"})

        with SessionLocal() as db:
            sandbox = db.get(SandboxEnvironment, env_name)
            if sandbox:
                sandbox.status = "building"
                db.commit()
        logger.info("[task:build_sandbox] DB status → building for %s", env_name)

        async def on_progress(artifact_name: str, _value) -> None:
            label = {
                "app_code":          "App Generator",
                "instrumented_code": "Telemetry Instrumentation",
                "state_bridge_code": "State Bridge",
                "policy_dsl":        "Policy Rules",
                "reward_fn_code":    "Reward Function",
            }.get(artifact_name, artifact_name)
            logger.info("[task:build_sandbox] agent done: %s (%s)", label, env_name)
            publish({"log": f"[agent] {label} — done ✓"})
            publish({"artifact": artifact_name, "status": "done"})

        logger.info("[task:build_sandbox] starting extraction (LLM pass 1) for %s", env_name)
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
        logger.info("[task:build_sandbox] extraction complete for %s", env_name)
        publish({"log": "[forge] extraction complete — starting 5 parallel agents…"})

        logger.info("[task:build_sandbox] starting 5 parallel agents for %s", env_name)
        orchestrator = EnvironmentOrchestrator(on_progress=on_progress)
        await orchestrator.run(
            env_name=env_name,
            description=description,
            compiler_input=compiler_input,
            policy_requirements=policy_requirements,
            reward_requirements=reward_requirements,
        )
        logger.info("[task:build_sandbox] all agents complete for %s", env_name)
        publish({"log": "[forge] all agents finished — building Docker image…"})

        envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
        app_dir = envs_root / env_name / "app"
        logger.info("[task:build_sandbox] building Docker image from %s", app_dir)
        runtime = ContainerRuntime()

        def _docker_ops() -> tuple[str, str, int]:
            image_tag = runtime.build(env_name, app_dir)
            container_id, port = runtime.run(env_name, image_tag)
            return image_tag, container_id, port

        image_tag, container_id, port = await loop.run_in_executor(None, _docker_ops)
        logger.info("[task:build_sandbox] container running — port=%s env_name=%s", port, env_name)
        publish({"log": f"[forge] container running on port {port} ✓"})

        with SessionLocal() as db:
            sandbox = db.get(SandboxEnvironment, env_name)
            if sandbox:
                sandbox.status = "running"
                sandbox.container_id = container_id
                sandbox.container_port = port
                sandbox.image_tag = image_tag
                db.commit()
        logger.info("[task:build_sandbox] DB status → running for %s", env_name)

    try:
        asyncio.run(_orchestrate())
        logger.info("[task:build_sandbox] COMPLETED — env_name=%s", env_name)
        publish({"done": True})
    except Exception as exc:
        logger.exception("[task:build_sandbox] FAILED — env_name=%s error=%s", env_name, exc)
        publish({"log": f"[forge] ERROR: {exc}"})
        with SessionLocal() as db:
            sandbox = db.get(SandboxEnvironment, env_name)
            if sandbox:
                sandbox.status = "error"
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

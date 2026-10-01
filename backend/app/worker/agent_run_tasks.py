"""Celery tasks that run an AgentRun's episodes against a containerized environment."""
from __future__ import annotations

import json
import logging
import secrets
from dataclasses import replace
from datetime import datetime, timezone

from celery import chain

from backend.app.models import AgentEpisode, AgentRun, SandboxEnvironment
from backend.app.worker.celery_app import celery
from backend.app.worker.container_episode import ContainerEpisode, browser_cdp_port
from backend.app.worker.env_lock import exclusive_environment
from backend.app.worker.run_records import count_finished_episode, mark_dispatch_failed
from forge.contracts.termination import is_budget_reason
from forge.settings import generated_envs_root, redis_url

logger = logging.getLogger("backend.app.worker.tasks")


@celery.task(bind=True, name="backend.app.worker.tasks.run_container_episode_task")
def run_container_episode_task(self, run_id: str, episode_index: int, seed: int) -> str:
    """Run a single agent episode against a containerized environment.

    The env-type specific work lives in `ContainerEpisode.run_attempt`.
    """
    from backend.app.database import get_session_factory

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
            count_finished_episode(AgentRun, run_id)
            return episode_id
        env_type = sb.env_type
        container_id = sb.container_id
        image_tag = sb.image_tag
        env_dir = generated_envs_root() / env_name
        jsonl_dir = env_dir / "agent_episodes"
        episode = ContainerEpisode(
            run_id=run_id,
            episode_id=episode_id,
            episode_index=episode_index,
            seed=seed,
            num_episodes=run.num_episodes,
            env_name=env_name,
            env_type=env_type,
            env_dir=env_dir,
            jsonl_path=jsonl_dir / f"{episode_id}.jsonl",
            agent_id=run.agent_id,
            objective=run.objective,
            max_steps=run.max_steps,
            dead_end_patience=run.dead_end_patience,
            success_threshold=run.success_threshold,
            container_port=sb.container_port,
        )

    jsonl_dir.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        db.add(AgentEpisode(
            id=episode_id,
            run_id=run_id,
            episode_index=episode_index,
            seed=seed,
            status="running",
            started_at=datetime.now(timezone.utc),
            jsonl_path=str(episode.jsonl_path),
        ))
        db.commit()

    import redis as _redis
    from forge.runtime.reliability import (
        execute_reliable_episode,
        compute_environment_version,
        classify_failure,
    )

    env_version = None
    try:
        env_version = compute_environment_version(
            env_name=env_name,
            env_type=env_type,
            image_id=image_tag,
            container_id=container_id,
            env_dir=env_dir,
        )

        with exclusive_environment(_redis.from_url(redis_url()), env_name):
            if env_type == "browser":
                episode = replace(episode, cdp_port=browser_cdp_port(container_id))

            with SessionLocal() as db_rel:
                result, attempts = execute_reliable_episode(
                    task_runner=episode.run_attempt,
                    task_id=f"{run_id}:{episode_index}",
                    env_name=env_name,
                    environment_version=env_version,
                    seed=seed,
                    db_session=db_rel,
                    logger=logger,
                )

            with SessionLocal() as db:
                ep = db.get(AgentEpisode, episode_id)
                if ep is not None:
                    ep.status = "truncated" if is_budget_reason(result.termination_reason) else "completed"
                    ep.total_steps = len(result.steps)
                    ep.total_reward = result.total_reward
                    ep.final_objective_score = result.final_objective_score
                    ep.termination_reason = result.termination_reason
                    ep.environment_version = env_version
                    ep.attempts_json = json.dumps([a.to_dict() for a in attempts])
                    ep.completed_at = datetime.now(timezone.utc)
                    db.commit()

    except Exception as exc:
        logger.exception("[container-ep] episode %s failed: %s", episode_id, exc)
        with SessionLocal() as db:
            ep = db.get(AgentEpisode, episode_id)
            if ep is not None:
                ftype, freason = classify_failure(exc)
                ep.status = "failed"
                ep.failure_type = ftype
                ep.failure_reason = freason
                ep.environment_version = env_version
                ep.termination_reason = str(exc)[:255]
                ep.completed_at = datetime.now(timezone.utc)
                db.commit()

    count_finished_episode(AgentRun, run_id)
    return episode_id

@celery.task(bind=True, name="backend.app.worker.tasks.run_container_run_task")
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
        env_name = run.env_name
        sandbox = db.get(SandboxEnvironment, env_name)
        cli_container = (
            sandbox.container_id if sandbox is not None and sandbox.env_type == "cli" else None
        )
        db.commit()

    try:
        if cli_container:
            from forge.envgen.container import ContainerRuntime
            # Freeze the shell (terminal setup included) once for the run, and
            # start the first episode's container now so it begins warm.
            runtime = ContainerRuntime()
            runtime.warm_cli(env_name, runtime.snapshot_cli(env_name, cli_container, run_id))
        # Every episode drives the environment's one container, so they run in
        # order. Immutable signatures keep each episode's return value out of
        # the next one's arguments.
        subtasks = chain(
            *(run_container_episode_task.si(run_id, i, seed_start + i)
              for i in range(num_episodes))
        )
        subtasks.apply_async()
    except Exception as exc:
        logger.exception("[container-run] dispatch failed for %s: %s", run_id, exc)
        mark_dispatch_failed(AgentRun, run_id, exc)

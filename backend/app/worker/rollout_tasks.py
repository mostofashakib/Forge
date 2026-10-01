"""Celery tasks that run a RolloutJob's episodes against in-process environments."""
from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone

from celery import group

from backend.app.models import Episode, RolloutJob
from backend.app.utils.env_loader import load_forge_env
from backend.app.worker.celery_app import celery
from backend.app.worker.run_records import count_finished_episode, mark_dispatch_failed
from forge.contracts.termination import is_budget_reason
from forge.settings import generated_envs_root

logger = logging.getLogger("backend.app.worker.tasks")


@celery.task(bind=True, name="backend.app.worker.tasks.run_episode_task")
def run_episode_task(self, rollout_job_id: str, episode_index: int, seed: int) -> str:
    """Run a single episode for a RolloutJob. Returns episode_id."""
    from backend.app.services.episode_collector import EpisodeDataCollector
    from backend.app.services.episode_service import create_episode
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
        create_episode(episode_id, env_name, task_name, seed, agent_id, db, jsonl_path=str(jsonl_path))

    with SessionLocal() as db_ep:
        from forge.runtime.reliability import (
            execute_reliable_episode,
            compute_environment_version,
            classify_failure,
        )

        env_version = compute_environment_version(env_name=env_name, env_dir=envs_root / env_name)

        def _execute_attempt(attempt: int, attempt_seed: int):
            telemetry = EpisodeDataCollector(
                episode_id=episode_id,
                db_session=db_ep,
                jsonl_path=jsonl_path,
            )
            env = load_forge_env(env_name, telemetry)
            try:
                selected_task = env.task_source.get(task_name)
            except KeyError:
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

            run_logger = AgentRunLogger(run_id=episode_id)
            trace_path = jsonl_path.with_name(f"{episode_id}.trace.jsonl")
            try:
                res = run_logged_episode(
                    env,
                    agent,
                    run_logger,
                    seed=attempt_seed,
                    task=selected_task,
                )
                return res
            finally:
                trace_path.write_text(run_logger.to_jsonl())

        try:
            result, attempts = execute_reliable_episode(
                task_runner=_execute_attempt,
                task_id=task_name,
                env_name=env_name,
                environment_version=env_version,
                seed=seed,
                db_session=db_ep,
                logger=logger,
            )
            ep = db_ep.get(Episode, episode_id)
            if ep is not None:
                ep.environment_version = env_version
                ep.attempts_json = json.dumps([a.to_dict() for a in attempts])
                if is_budget_reason(getattr(result, "termination_reason", None)):
                    ep.status = "truncated"
                db_ep.commit()
        except Exception as exc:
            logger.exception("Episode %s failed: %s", episode_id, exc)
            ep = db_ep.get(Episode, episode_id)
            if ep is not None:
                ftype, freason = classify_failure(exc)
                ep.status = "failed"
                ep.failure_type = ftype
                ep.failure_reason = freason
                ep.environment_version = env_version
                ep.completed_at = datetime.now(timezone.utc)
                db_ep.commit()

    count_finished_episode(RolloutJob, rollout_job_id)
    return episode_id

@celery.task(bind=True, name="backend.app.worker.tasks.run_rollout_task")
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
        mark_dispatch_failed(RolloutJob, rollout_job_id, exc)

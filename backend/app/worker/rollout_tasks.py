"""Celery tasks that run a RolloutJob's episodes against in-process environments."""
from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from celery import group

from backend.app.models import Episode, RolloutJob
from backend.app.utils.env_loader import forge_env_builder, load_forge_env
from backend.app.worker.celery_app import celery
from backend.app.worker.run_records import count_finished_episode, mark_dispatch_failed
from forge.contracts.termination import is_budget_reason
from forge.runtime.reliability.failures import FAILURE_TYPE_CONFIGURATION
from forge.settings import generated_envs_root

logger = logging.getLogger("backend.app.worker.tasks")


@celery.task(bind=True, name="backend.app.worker.tasks.run_episode_task")
def run_episode_task(self, rollout_job_id: str, episode_index: int, seed: int) -> str:
    """Run a single episode for a RolloutJob. Returns episode_id."""
    from backend.app.services.episode_service import create_episode
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

    unknown_task = _unknown_task(env_name, task_name)
    if unknown_task is not None:
        # Retrying cannot fix a task name, and nothing about the environment
        # failed, so the episode fails once without retries or quarantine.
        logger.error("Episode %s not run: %s", episode_id, unknown_task)
        _fail_episode(SessionLocal, episode_id, FAILURE_TYPE_CONFIGURATION, str(unknown_task))
        count_finished_episode(RolloutJob, rollout_job_id)
        return episode_id

    with SessionLocal() as db_ep:
        from forge.runtime.reliability import (
            execute_reliable_episode,
            compute_environment_version,
            classify_failure,
        )

        env_version = compute_environment_version(env_name=env_name, env_dir=envs_root / env_name)
        attempt = _RolloutAttempt(
            episode_id=episode_id, env_name=env_name, task_name=task_name,
            agent_id=agent_id, jsonl_path=jsonl_path, db_session=db_ep,
        )
        try:
            result, attempts = execute_reliable_episode(
                task_runner=attempt,
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

@dataclass(frozen=True)
class _RolloutAttempt:
    """One attempt of a rollout episode: a fresh env, the job's agent, and a trace."""

    episode_id: str
    env_name: str
    task_name: str
    agent_id: str
    jsonl_path: Path
    db_session: object

    def __call__(self, attempt: int, attempt_seed: int):
        from backend.app.services.episode_collector import EpisodeDataCollector
        from forge.runtime.agent_logger import AgentRunLogger, run_logged_episode
        from forge.runtime.agents.factory import make_agent

        telemetry = EpisodeDataCollector(
            episode_id=self.episode_id,
            db_session=self.db_session,
            jsonl_path=self.jsonl_path,
        )
        env = load_forge_env(self.env_name, telemetry)
        task = _select_task(env, self.task_name)
        agent = make_agent(self.agent_id, environment=env, task=task)
        run_logger = AgentRunLogger(run_id=self.episode_id)
        try:
            return run_logged_episode(env, agent, run_logger, seed=attempt_seed, task=task)
        finally:
            self.jsonl_path.with_name(f"{self.episode_id}.trace.jsonl").write_text(run_logger.to_jsonl())


class UnknownTaskError(LookupError):
    """The rollout names a task its environment does not declare."""

    def __init__(self, task_name: str, available: list[str]) -> None:
        super().__init__(
            f"unknown_task: environment has no task {task_name!r}; declared tasks: {', '.join(available)}"
        )
        self.task_name = task_name
        self.available = available


def _select_task(env, task_name: str):
    """The declared task named `task_name`. Never another task in its place.

    An environment that declares no tasks runs the name as a free-form
    objective, which is exactly what was asked for.
    """
    declared = env.task_source.tasks()
    if not declared:
        return {"id": task_name, "objective": task_name}
    try:
        return env.task_source.get(task_name)
    except KeyError:
        raise UnknownTaskError(task_name, [task.id for task in declared]) from None


def _unknown_task(env_name: str, task_name: str) -> UnknownTaskError | None:
    """The error when `env_name` does not declare `task_name`, else None.

    Any other failure to build the environment is left to the attempt, where
    the reliability executor classifies and retries it.
    """
    try:
        env = forge_env_builder(env_name)()
    except Exception as exc:
        logger.warning("could not check task %r before running %s: %s", task_name, env_name, exc)
        return None
    try:
        _select_task(env, task_name)
    except UnknownTaskError as exc:
        return exc
    return None


def _fail_episode(session_factory, episode_id: str, failure_type: str, reason: str) -> None:
    with session_factory() as db:
        episode = db.get(Episode, episode_id)
        if episode is None:
            return
        episode.status = "failed"
        episode.failure_type = failure_type
        episode.failure_reason = reason
        episode.completed_at = datetime.now(timezone.utc)
        db.commit()


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

from __future__ import annotations
import importlib
import logging
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

from celery import group
from sqlalchemy import update

from backend.app.worker.celery_app import celery
from backend.app.models import RolloutJob, Episode

logger = logging.getLogger(__name__)


def _load_forge_env(env_name: str, telemetry):
    """Load a generated ForgeEnv and inject telemetry."""
    envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
    parent = str(envs_root.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(f"generated_envs.{env_name}.gym_wrapper")
    build_fn = getattr(module, f"build_{env_name}_env")
    env = build_fn()
    env._telemetry = telemetry
    return env


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
            env = _load_forge_env(env_name, telemetry)
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


@celery.task(bind=True)
def run_rollout_task(self, rollout_job_id: str) -> None:
    """Dispatch all episode subtasks for a RolloutJob."""
    from backend.app.database import get_session_factory

    SessionLocal = get_session_factory()

    with SessionLocal() as db:
        job = db.get(RolloutJob, rollout_job_id)
        if job is None:
            logger.error("RolloutJob %s not found", rollout_job_id)
            return
        job.status = "running"
        num_episodes = job.num_episodes
        seed_start = job.seed_start
        db.commit()

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

from __future__ import annotations
import secrets
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import RolloutJob
from backend.app.worker.tasks import run_rollout_task


def create_rollout(
    env_name: str,
    task_name: str,
    agent_id: str,
    num_episodes: int,
    seed_start: int,
    db: Session,
) -> RolloutJob:
    job_id = f"rj_{secrets.token_hex(4)}"
    job = RolloutJob(
        id=job_id,
        env_name=env_name,
        task_name=task_name,
        agent_id=agent_id,
        num_episodes=num_episodes,
        seed_start=seed_start,
        status="pending",
        episodes_completed=0,
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    run_rollout_task.apply_async(args=[job_id])
    return job


def get_rollout(rollout_job_id: str, db: Session) -> RolloutJob | None:
    return db.get(RolloutJob, rollout_job_id)


def list_rollouts(env_name: str, db: Session) -> list[RolloutJob]:
    return list(
        db.execute(
            select(RolloutJob)
            .where(RolloutJob.env_name == env_name)
            .order_by(RolloutJob.created_at.desc())
        ).scalars()
    )

# backend/app/services/episode_service.py
from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from backend.app.models import Episode, EpisodeStep
from backend.app.services.episode_stats import get_stats as get_stats

__all__ = ["get_stats"]


def create_episode(
    episode_id: str,
    env_name: str,
    task_name: str,
    seed: int,
    agent_id: str,
    db: Session,
    jsonl_path: str | None = None,
) -> Episode:
    ep = Episode(
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
        jsonl_path=jsonl_path,
    )
    db.add(ep)
    db.commit()
    return ep


def get_episode(episode_id: str, db: Session) -> Episode | None:
    return db.get(Episode, episode_id)


def get_episode_steps(episode_id: str, db: Session) -> list[EpisodeStep]:
    return (
        db.query(EpisodeStep)
        .filter_by(episode_id=episode_id)
        .order_by(EpisodeStep.step_index)
        .all()
    )


def list_episodes(env_name: str, db: Session, limit: int = 20) -> list[Episode]:
    return (
        db.query(Episode)
        .filter_by(env_name=env_name)
        .order_by(Episode.started_at.desc())
        .limit(limit)
        .all()
    )


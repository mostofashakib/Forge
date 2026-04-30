from __future__ import annotations
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.app.models import Episode, EpisodeStep


def get_episodes(env_name: str, db: Session) -> list[Episode]:
    return list(
        db.execute(
            select(Episode)
            .where(Episode.env_name == env_name, Episode.status == "completed")
            .order_by(Episode.started_at)
        ).scalars()
    )


def get_steps(episode_id: str, db: Session) -> list[EpisodeStep]:
    return list(
        db.execute(
            select(EpisodeStep)
            .where(EpisodeStep.episode_id == episode_id)
            .order_by(EpisodeStep.step_index)
        ).scalars()
    )

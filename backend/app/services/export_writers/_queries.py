from __future__ import annotations
from collections import defaultdict
from sqlalchemy import select
from sqlalchemy.orm import Session, defer
from backend.app.models import Episode, EpisodeStep


def _completed(env_name: str):
    return (Episode.env_name == env_name, Episode.status == "completed")


def get_episodes(env_name: str, db: Session) -> list[Episode]:
    return list(
        db.execute(
            select(Episode)
            .where(*_completed(env_name))
            .order_by(Episode.started_at)
        ).scalars()
    )


def get_steps_by_episode(env_name: str, db: Session) -> dict[str, list[EpisodeStep]]:
    """Every step of the env's completed episodes, in one query, keyed by episode.

    No writer reads `diff` or `events`, so those blobs stay in the database.
    """
    steps = db.execute(
        select(EpisodeStep)
        .join(Episode, Episode.id == EpisodeStep.episode_id)
        .where(*_completed(env_name))
        .options(defer(EpisodeStep.diff), defer(EpisodeStep.events))
        .order_by(EpisodeStep.episode_id, EpisodeStep.step_index)
    ).scalars()
    grouped: dict[str, list[EpisodeStep]] = defaultdict(list)
    for step in steps:
        grouped[step.episode_id].append(step)
    return grouped

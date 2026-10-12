"""The recent agent episodes that evaluation and anomaly detection grade."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from backend.app.models import AgentEpisode, AgentRun
from backend.app.services.trajectory_files import read_trajectory

logger = logging.getLogger(__name__)

_RECENT_RUNS = 10


@dataclass(frozen=True)
class EpisodeSample:
    episodes: list[AgentEpisode]
    latest_objective: str | None


def recent_completed_episodes(
    env_name: str,
    db: Session,
    *,
    limit: int,
    oldest_first: bool = False,
) -> EpisodeSample:
    """Completed episodes from the env's most recent runs, ordered by completion."""
    runs = (
        db.query(AgentRun)
        .filter(AgentRun.env_name == env_name)
        .order_by(AgentRun.created_at.desc())
        .limit(_RECENT_RUNS)
        .all()
    )
    completed_at = AgentEpisode.completed_at
    episodes = (
        db.query(AgentEpisode)
        .filter(
            AgentEpisode.run_id.in_([run.id for run in runs]),
            AgentEpisode.status == "completed",
        )
        .order_by(completed_at.asc() if oldest_first else completed_at.desc())
        .limit(limit)
        .all()
    )
    return EpisodeSample(
        episodes=episodes,
        latest_objective=runs[0].objective if runs else None,
    )


def load_trajectory_steps(episode: AgentEpisode, *, max_steps: int) -> list[dict]:
    """The last `max_steps` recorded steps of an episode, without its summary line."""
    if not episode.jsonl_path:
        return []
    path = Path(episode.jsonl_path)
    try:
        return read_trajectory(path, max_steps=max_steps).steps
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        logger.warning("[episode-sample] unreadable trajectory %s: %s", path, exc)
        return []

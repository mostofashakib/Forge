# forge/runtime/replay.py
from __future__ import annotations
import json
from dataclasses import dataclass
from sqlalchemy.orm import Session
from backend.app.models import Episode, EpisodeStep


@dataclass
class EpisodeRecord:
    episode: Episode
    steps: list[EpisodeStep]


class ReplayService:
    def load_episode(self, episode_id: str, db: Session) -> EpisodeRecord:
        ep = db.get(Episode, episode_id)
        steps = (
            db.query(EpisodeStep)
            .filter_by(episode_id=episode_id)
            .order_by(EpisodeStep.step_index)
            .all()
        )
        return EpisodeRecord(episode=ep, steps=steps)

    def branch_from(self, episode_id: str, step_n: int, db: Session) -> list[dict]:
        steps = (
            db.query(EpisodeStep)
            .filter(
                EpisodeStep.episode_id == episode_id,
                EpisodeStep.step_index < step_n,
            )
            .order_by(EpisodeStep.step_index)
            .all()
        )
        return [json.loads(s.action) for s in steps]

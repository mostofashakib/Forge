from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy.orm import Session

from ._queries import get_episodes, get_steps_by_episode
from .files import write_jsonl
from .gate import DatasetGate


def write(env_name: str, db: Session, out_dir: Path, gate: DatasetGate) -> None:
    write_jsonl(out_dir / "rewards.jsonl", gate.admitted(_rows(env_name, db), "rewards"))


def _rows(env_name: str, db: Session) -> Iterator[dict]:
    steps_by_episode = get_steps_by_episode(env_name, db)
    for ep in get_episodes(env_name, db):
        yield {
            "episode_id": ep.id,
            "total_reward": ep.total_reward,
            "passed": ep.passed,
            "components": [
                {"step_index": s.step_index, "reward": s.reward}
                for s in steps_by_episode.get(ep.id, [])
            ],
        }

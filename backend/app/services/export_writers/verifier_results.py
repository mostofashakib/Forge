from __future__ import annotations
import json
from pathlib import Path
from sqlalchemy.orm import Session
from ._queries import get_episodes, get_steps_by_episode


def write(env_name: str, db: Session, out_dir: Path) -> None:
    episodes = get_episodes(env_name, db)
    steps_by_episode = get_steps_by_episode(env_name, db)
    with (out_dir / "verifier_results.jsonl").open("w") as f:
        for ep in episodes:
            steps = steps_by_episode.get(ep.id, [])
            for s in steps:
                record = {
                    "episode_id": ep.id,
                    "step_index": s.step_index,
                    "results": json.loads(s.verifier_results),
                }
                f.write(json.dumps(record) + "\n")

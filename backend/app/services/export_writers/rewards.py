from __future__ import annotations
import json
from pathlib import Path
from sqlalchemy.orm import Session
from ._queries import get_episodes, get_steps


def write(env_name: str, db: Session, out_dir: Path) -> None:
    episodes = get_episodes(env_name, db)
    with (out_dir / "rewards.jsonl").open("w") as f:
        for ep in episodes:
            steps = get_steps(ep.id, db)
            record = {
                "episode_id": ep.id,
                "total_reward": ep.total_reward,
                "passed": ep.passed,
                "components": [
                    {"step_index": s.step_index, "reward": s.reward}
                    for s in steps
                ],
            }
            f.write(json.dumps(record) + "\n")

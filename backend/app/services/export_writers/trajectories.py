from __future__ import annotations
import json
from pathlib import Path
from sqlalchemy.orm import Session
from ._queries import get_episodes, get_steps


def write(env_name: str, db: Session, out_dir: Path) -> None:
    episodes = get_episodes(env_name, db)
    with (out_dir / "trajectories.jsonl").open("w") as f:
        for ep in episodes:
            steps = get_steps(ep.id, db)
            record = {
                "episode_id": ep.id,
                "env": ep.env_name,
                "task": ep.task_name,
                "seed": ep.seed,
                "steps": [
                    {
                        "step_index": s.step_index,
                        "action": json.loads(s.action),
                        "reward": s.reward,
                        "terminated": s.terminated,
                        "truncated": s.truncated,
                    }
                    for s in steps
                ],
            }
            f.write(json.dumps(record) + "\n")

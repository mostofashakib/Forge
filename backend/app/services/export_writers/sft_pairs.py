from __future__ import annotations
import json
from pathlib import Path
from sqlalchemy.orm import Session
from ._queries import get_episodes, get_steps


def write(env_name: str, db: Session, out_dir: Path) -> None:
    episodes = get_episodes(env_name, db)
    with (out_dir / "sft_pairs.jsonl").open("w") as f:
        for ep in episodes:
            if not ep.passed:
                continue
            steps = get_steps(ep.id, db)
            for s in steps:
                record = {
                    "prompt": json.dumps({"step_index": s.step_index}),
                    "completion": s.action,
                }
                f.write(json.dumps(record) + "\n")

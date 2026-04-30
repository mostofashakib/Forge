from __future__ import annotations
from pathlib import Path
from sqlalchemy.orm import Session
import pandas as pd
from ._queries import get_episodes


def write(env_name: str, db: Session, out_dir: Path) -> None:
    episodes = get_episodes(env_name, db)
    rows = [
        {
            "episode_id": ep.id,
            "env_name": ep.env_name,
            "task_name": ep.task_name,
            "seed": ep.seed,
            "agent_id": ep.agent_id,
            "total_reward": ep.total_reward,
            "passed": ep.passed,
            "total_steps": ep.total_steps,
        }
        for ep in episodes
    ]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["episode_id", "env_name", "task_name", "seed", "agent_id", "total_reward", "passed", "total_steps"]
    )
    df.to_parquet(out_dir / "grpo_rollouts.parquet", index=False)

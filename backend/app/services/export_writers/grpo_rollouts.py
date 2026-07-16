from __future__ import annotations
import json
from pathlib import Path
from sqlalchemy.orm import Session
import pandas as pd
from ._queries import get_episodes, get_steps
from .common import action_to_command


def write(env_name: str, db: Session, out_dir: Path) -> None:
    """RL rollout table for GRPO / PPO training.

    Each row is one completed episode with:
      - prompt / completion strings for the policy model
      - total_reward and per_step_rewards for the reward model
      - episode metadata for filtering and grouping
    Compatible with TRL GRPOTrainer and veRL.
    """
    episodes = get_episodes(env_name, db)
    rows = []
    for ep in episodes:
        steps = get_steps(ep.id, db)
        commands = [action_to_command(s.action) for s in steps]
        per_step_rewards = [s.reward for s in steps]
        rows.append({
            "episode_id": ep.id,
            "env_name": ep.env_name,
            "task_name": ep.task_name,
            "seed": ep.seed,
            "agent_id": ep.agent_id,
            "prompt": f"Task: {ep.task_name}\nEnvironment: {ep.env_name}",
            "completion": "\n".join(f"$ {c}" for c in commands),
            "total_reward": ep.total_reward,
            "passed": ep.passed,
            "total_steps": ep.total_steps,
            "per_step_rewards": json.dumps(per_step_rewards),
        })

    cols = [
        "episode_id", "env_name", "task_name", "seed", "agent_id",
        "prompt", "completion", "total_reward", "passed", "total_steps",
        "per_step_rewards",
    ]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
    df.to_parquet(out_dir / "grpo_rollouts.parquet", index=False)

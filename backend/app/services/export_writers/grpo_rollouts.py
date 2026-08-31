from __future__ import annotations
import json
from pathlib import Path
from sqlalchemy.orm import Session
import pandas as pd
from forge.contracts import RolloutRecord
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
        final_verification = _verification_results(steps[-1]) if steps else []
        reward_breakdown = {
            "total_reward": ep.total_reward,
            "components": [
                {"name": "authoritative_episode_reward", "value": ep.total_reward}
            ],
        }
        record = RolloutRecord(
            episode_id=ep.id,
            env_name=ep.env_name,
            task_name=ep.task_name,
            seed=ep.seed,
            prompt=f"Task: {ep.task_name}\nEnvironment: {ep.env_name}",
            completion="\n".join(f"$ {c}" for c in commands),
            total_reward=ep.total_reward,
            passed=ep.passed,
            outcome="success" if ep.passed else "failure",
            steps=ep.total_steps,
            per_step_rewards=per_step_rewards,
            behavior_model=ep.agent_id,
            termination_reason=ep.termination_reason,
            verification_results=final_verification,
            reward_breakdown=reward_breakdown,
        )
        row = record.model_dump()
        row["agent_id"] = ep.agent_id
        row["total_steps"] = ep.total_steps
        row["per_step_rewards"] = json.dumps(per_step_rewards)
        row["verification_results"] = json.dumps(row["verification_results"])
        row["reward_breakdown"] = json.dumps(row["reward_breakdown"])
        rows.append(row)

    cols = [
        "episode_id", "env_name", "task_name", "seed", "agent_id",
        "prompt", "completion", "total_reward", "passed", "total_steps",
        "per_step_rewards", "behavior_model", "termination_reason",
        "verification_results", "reward_breakdown",
    ]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
    df.to_parquet(out_dir / "grpo_rollouts.parquet", index=False)


def _verification_results(step) -> list[dict]:
    try:
        value = json.loads(step.verifier_results)
    except (TypeError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []

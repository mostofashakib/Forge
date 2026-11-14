from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from forge.contracts.dataset_rows import RolloutRow, actions_text

from ._queries import get_episodes, get_steps_by_episode
from .files import stored_rollout, write_rollouts
from .gate import DatasetGate


def write(env_name: str, db: Session, out_dir: Path, gate: DatasetGate) -> None:
    """RL rollout table for GRPO / PPO training.

    Each row is one completed episode with:
      - prompt / completion strings for the policy model
      - total_reward and per_step_rewards for the reward model
      - episode metadata for filtering and grouping
    Compatible with TRL GRPOTrainer and veRL.
    """
    steps_by_episode = get_steps_by_episode(env_name, db)
    rows = (
        stored_rollout(_row(ep, steps_by_episode.get(ep.id, [])))
        for ep in get_episodes(env_name, db)
    )
    write_rollouts(out_dir / "grpo_rollouts.parquet", list(gate.admitted(rows, "grpo_rollouts")))


def _row(ep, steps) -> dict:
    return RolloutRow(
        episode_id=ep.id,
        env_name=ep.env_name,
        task_name=ep.task_name,
        seed=ep.seed,
        prompt=f"Task: {ep.task_name}\nEnvironment: {ep.env_name}",
        completion=actions_text([s.action for s in steps]),
        total_reward=ep.total_reward,
        passed=ep.passed,
        outcome="success" if ep.passed else "failure",
        steps=ep.total_steps,
        per_step_rewards=[s.reward for s in steps],
        behavior_model=ep.agent_id,
        termination_reason=ep.termination_reason,
        verification_results=_verification_results(steps[-1]) if steps else [],
        reward_breakdown={
            "total_reward": ep.total_reward,
            "components": [{"name": "authoritative_episode_reward", "value": ep.total_reward}],
        },
        agent_id=ep.agent_id,
        total_steps=ep.total_steps,
    ).model_dump()


def _verification_results(step) -> list[dict]:
    try:
        value = json.loads(step.verifier_results)
    except (TypeError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []

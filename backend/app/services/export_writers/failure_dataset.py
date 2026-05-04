from __future__ import annotations
import json
from pathlib import Path
from sqlalchemy.orm import Session
from ._queries import get_episodes, get_steps


def _action_to_command(action_raw: str) -> str:
    try:
        action = json.loads(action_raw) if action_raw else {}
    except (json.JSONDecodeError, TypeError):
        return str(action_raw)
    return action.get("command") or action.get("cmd") or json.dumps(action)


def write(env_name: str, db: Session, out_dir: Path) -> None:
    """Failure dataset — full trajectories from episodes that did not pass.

    Each record includes step-level actions, rewards, and verifier diagnostics
    so failure modes can be studied or used for contrastive/adversarial training.
    """
    episodes = get_episodes(env_name, db)
    with (out_dir / "failure_dataset.jsonl").open("w") as f:
        for ep in episodes:
            if ep.passed:
                continue
            steps = get_steps(ep.id, db)
            step_records = []
            for s in steps:
                try:
                    action = json.loads(s.action) if s.action else {}
                except (json.JSONDecodeError, TypeError):
                    action = {}
                try:
                    verifier = json.loads(s.verifier_results) if s.verifier_results else []
                except (json.JSONDecodeError, TypeError):
                    verifier = []

                step_records.append({
                    "step_index": s.step_index,
                    "command": _action_to_command(s.action),
                    "action": action,
                    "reward": s.reward,
                    "verifier_results": verifier,
                    "terminated": s.terminated,
                    "truncated": s.truncated,
                })

            record = {
                "episode_id": ep.id,
                "env_name": ep.env_name,
                "task_name": ep.task_name,
                "seed": ep.seed,
                "total_reward": ep.total_reward,
                "total_steps": ep.total_steps,
                "steps": step_records,
            }
            f.write(json.dumps(record) + "\n")

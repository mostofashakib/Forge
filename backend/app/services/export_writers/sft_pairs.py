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
    """SFT pairs from successful episodes only.

    Format: messages array with user (task) and assistant (command sequence).
    Compatible with OpenAI fine-tuning, TRL SFTTrainer, and Axolotl.
    """
    episodes = get_episodes(env_name, db)
    with (out_dir / "sft_pairs.jsonl").open("w") as f:
        for ep in episodes:
            if not ep.passed:
                continue
            steps = get_steps(ep.id, db)
            if not steps:
                continue
            commands = [_action_to_command(s.action) for s in steps]
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Task: {ep.task_name}\nEnvironment: {ep.env_name}",
                    },
                    {
                        "role": "assistant",
                        "content": "\n".join(f"$ {c}" for c in commands),
                    },
                ],
                "episode_id": ep.id,
                "total_reward": ep.total_reward,
                "total_steps": ep.total_steps,
            }
            f.write(json.dumps(record) + "\n")

from __future__ import annotations
import json
from pathlib import Path
from sqlalchemy.orm import Session
from ._queries import get_episodes, get_steps
from .common import action_to_command


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
            command_history: list[str] = []
            for step in steps:
                command = action_to_command(step.action)
                context = f"Task: {ep.task_name}\nEnvironment: {ep.env_name}"
                if command_history:
                    context += "\nPrevious actions:\n" + "\n".join(
                        f"$ {previous}" for previous in command_history
                    )
                record = {
                    "messages": [
                        {"role": "user", "content": context},
                        {"role": "assistant", "content": f"$ {command}"},
                    ],
                    "episode_id": ep.id,
                    "step_index": step.step_index,
                    "total_reward": ep.total_reward,
                    "step_reward": step.reward,
                }
                f.write(json.dumps(record) + "\n")
                command_history.append(command)

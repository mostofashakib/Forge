from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy.orm import Session

from forge.contracts.dataset_rows import SftStep, action_line, step_exchanges

from ._queries import get_episodes, get_steps_by_episode
from .files import write_jsonl
from .gate import DatasetGate


class EpisodeSftStep(SftStep):
    step_reward: float


def write(env_name: str, db: Session, out_dir: Path, gate: DatasetGate) -> None:
    """SFT pairs from successful episodes only.

    Format: messages array with user (task) and assistant (command sequence).
    Compatible with OpenAI fine-tuning, TRL SFTTrainer, and Axolotl.
    """
    write_jsonl(out_dir / "sft_pairs.jsonl", gate.admitted(_rows(env_name, db), "sft_pairs"))


def _rows(env_name: str, db: Session) -> Iterator[dict]:
    steps_by_episode = get_steps_by_episode(env_name, db)
    for ep in get_episodes(env_name, db):
        if not ep.passed:
            continue
        steps = steps_by_episode.get(ep.id, [])
        prompt = f"Task: {ep.task_name}\nEnvironment: {ep.env_name}"
        for step, messages in zip(steps, step_exchanges(prompt, [action_line(s.action) for s in steps])):
            yield EpisodeSftStep(
                messages=messages, episode_id=ep.id, step_index=step.step_index,
                total_reward=ep.total_reward, step_reward=step.reward,
            ).model_dump()

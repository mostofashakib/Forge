from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy.orm import Session

from forge.contracts.dataset_rows import FailureEpisode, action_command

from ._queries import get_episodes, get_steps_by_episode
from .files import write_jsonl
from .gate import DatasetGate


class EpisodeFailure(FailureEpisode):
    seed: int


def write(env_name: str, db: Session, out_dir: Path, gate: DatasetGate) -> None:
    """Failure dataset — full trajectories from episodes that did not pass.

    Each record includes step-level actions, rewards, and verifier diagnostics
    so failure modes can be studied or used for contrastive/adversarial training.
    """
    write_jsonl(out_dir / "failure_dataset.jsonl", gate.admitted(_rows(env_name, db), "failure_dataset"))


def _rows(env_name: str, db: Session) -> Iterator[dict]:
    steps_by_episode = get_steps_by_episode(env_name, db)
    for ep in get_episodes(env_name, db):
        if ep.passed:
            continue
        yield EpisodeFailure(
            episode_id=ep.id,
            env_name=ep.env_name,
            task_name=ep.task_name,
            seed=ep.seed,
            total_reward=ep.total_reward,
            total_steps=ep.total_steps,
            steps=[_step(s) for s in steps_by_episode.get(ep.id, [])],
        ).model_dump()


def _step(s) -> dict:
    return {
        "step_index": s.step_index,
        "command": action_command(s.action),
        "action": _json(s.action, {}),
        "reward": s.reward,
        "verifier_results": _json(s.verifier_results, []),
        "terminated": s.terminated,
        "truncated": s.truncated,
    }


def _json(raw: str, empty):
    try:
        return json.loads(raw) if raw else empty
    except (json.JSONDecodeError, TypeError):
        return empty

"""Stage training data from what Forge actually recorded.

Every row written here comes from a graded episode or a validated golden
trajectory. Nothing is padded or invented to make a run start: when no real
data exists, staging reports that and the caller refuses the run.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.models import Episode, TaskBatch
from backend.app.services import task_registry

logger = logging.getLogger(__name__)


def episode_counts_by_env(db: Session) -> dict[str, int]:
    """Episode count per environment, in one grouped query."""
    rows = db.query(Episode.env_name, func.count(Episode.id)).group_by(Episode.env_name).all()
    return dict(rows)


def stage_available_data(db: Session, data_dir: Path) -> bool:
    """Stage the first real source found: graded episodes, then the latest batch.

    Returns False, writing nothing, when neither exists.
    """
    return _stage_episode_exports(db, data_dir) or _stage_latest_batch(db, data_dir)


def _stage_episode_exports(db: Session, data_dir: Path) -> bool:
    from backend.app.services.export_writers import grpo_rollouts, preference_pairs, sft_pairs
    from backend.app.services.task_factory_targets import list_targets

    episode_counts = episode_counts_by_env(db)
    for target in list_targets(db):
        if episode_counts.get(target.name):
            for writer in (grpo_rollouts, preference_pairs, sft_pairs):
                writer.write(target.name, db, data_dir)
            return True
    return False


def _stage_latest_batch(db: Session, data_dir: Path) -> bool:
    batch = (
        db.query(TaskBatch)
        .filter(TaskBatch.deleted_at.is_(None))
        .order_by(TaskBatch.created_at.desc())
        .first()
    )
    if batch is None:
        return False
    detail = task_registry.get_batch(db, batch.id)
    if not detail:
        return False
    stage_generator_batch(detail, data_dir)
    return True


def stage_batch_by_id(db: Session, batch_id: str, target_dir: Path) -> Path:
    """Stage one saved batch into `target_dir`. A missing batch stages nothing."""
    target_dir.mkdir(parents=True, exist_ok=True)
    detail = task_registry.get_batch(db, batch_id)
    if detail:
        stage_generator_batch(detail, target_dir)
    return target_dir


def stage_generator_batch(batch: dict, target_dir: Path) -> Path:
    """Write a batch's validated golden trajectories as SFT pairs and GRPO rollouts.

    Preference pairs are not written: a batch holds no rejected answer to an
    accepted task, only rejected tasks, so any pair would be made up.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    env_name = batch.get("env_name", "generator")
    (target_dir / "batch_export.json").write_text(json.dumps(batch, indent=2))

    trajectories = [
        (task, _prompt(task, env_name))
        for task in batch.get("tasks", [])
        if task.get("objective") and task.get("golden")
    ]
    if not trajectories:
        return target_dir
    _write_jsonl(target_dir / "sft_pairs.jsonl", [_sft_row(task, prompt) for task, prompt in trajectories])
    _write_rollouts(
        target_dir / "grpo_rollouts.parquet",
        [_rollout_row(task, prompt, env_name, index) for index, (task, prompt) in enumerate(trajectories)],
    )
    return target_dir


def _prompt(task: dict, env_name: str) -> str:
    return f"Task: {task['objective']}\nEnvironment: {env_name}"


def _step_text(step) -> str:
    return (step.get("command") or step.get("tool", "")) if isinstance(step, dict) else str(step)


def _sft_row(task: dict, prompt: str) -> dict:
    completion = "\n".join(f"$ {_step_text(step)}" for step in task["golden"])
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "prompt": prompt,
        "completion": completion,
        "total_reward": 1.0,
    }


def _rollout_row(task: dict, prompt: str, env_name: str, index: int) -> dict:
    # Golden trajectories reproduced their final state on every pass^k run,
    # so they are recorded as passing episodes.
    return {
        "episode_id": f"ep_gen_{task.get('id', index)}",
        "env_name": env_name,
        "task_name": task["objective"][:40],
        "prompt": prompt,
        "completion": "\n".join(_step_text(step) for step in task["golden"]),
        "total_reward": 1.0,
        "passed": True,
        "per_step_rewards": json.dumps([1.0]),
        "behavior_model": "synthetic_generator",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_rollouts(path: Path, rows: list[dict]) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_parquet(path, index=False)

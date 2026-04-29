from __future__ import annotations
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import Episode, EpisodeStep, ExportJob

logger = logging.getLogger(__name__)
BASE_DIR = Path("generated_envs")


def _get_episodes(env_name: str, db: Session) -> list[Episode]:
    return list(
        db.execute(
            select(Episode)
            .where(Episode.env_name == env_name, Episode.status == "completed")
            .order_by(Episode.started_at)
        ).scalars()
    )


def _get_steps(episode_id: str, db: Session) -> list[EpisodeStep]:
    return list(
        db.execute(
            select(EpisodeStep)
            .where(EpisodeStep.episode_id == episode_id)
            .order_by(EpisodeStep.step_index)
        ).scalars()
    )


def _write_trajectories(env_name: str, db: Session, out_dir: Path) -> None:
    episodes = _get_episodes(env_name, db)
    with (out_dir / "trajectories.jsonl").open("w") as f:
        for ep in episodes:
            steps = _get_steps(ep.id, db)
            record = {
                "episode_id": ep.id,
                "env": ep.env_name,
                "task": ep.task_name,
                "seed": ep.seed,
                "steps": [
                    {
                        "step_index": s.step_index,
                        "action": json.loads(s.action),
                        "reward": s.reward,
                        "terminated": s.terminated,
                        "truncated": s.truncated,
                    }
                    for s in steps
                ],
            }
            f.write(json.dumps(record) + "\n")


def _write_rewards(env_name: str, db: Session, out_dir: Path) -> None:
    episodes = _get_episodes(env_name, db)
    with (out_dir / "rewards.jsonl").open("w") as f:
        for ep in episodes:
            steps = _get_steps(ep.id, db)
            components = [
                {"step_index": s.step_index, "reward": s.reward}
                for s in steps
            ]
            record = {
                "episode_id": ep.id,
                "total_reward": ep.total_reward,
                "passed": ep.passed,
                "components": components,
            }
            f.write(json.dumps(record) + "\n")


def _write_verifier_results(env_name: str, db: Session, out_dir: Path) -> None:
    episodes = _get_episodes(env_name, db)
    with (out_dir / "verifier_results.jsonl").open("w") as f:
        for ep in episodes:
            steps = _get_steps(ep.id, db)
            for s in steps:
                record = {
                    "episode_id": ep.id,
                    "step_index": s.step_index,
                    "results": json.loads(s.verifier_results),
                }
                f.write(json.dumps(record) + "\n")


def _write_sft_pairs(env_name: str, db: Session, out_dir: Path) -> None:
    episodes = _get_episodes(env_name, db)
    with (out_dir / "sft_pairs.jsonl").open("w") as f:
        for ep in episodes:
            if not ep.passed:
                continue
            steps = _get_steps(ep.id, db)
            for s in steps:
                record = {
                    "prompt": json.dumps({"step_index": s.step_index}),
                    "completion": s.action,
                }
                f.write(json.dumps(record) + "\n")


def _write_preference_pairs(env_name: str, db: Session, out_dir: Path) -> None:
    episodes = _get_episodes(env_name, db)
    buckets: dict[tuple[str, int], list[Episode]] = defaultdict(list)
    for ep in episodes:
        key = (ep.task_name, ep.seed // 10)
        buckets[key].append(ep)

    with (out_dir / "preference_pairs.jsonl").open("w") as f:
        for (task, bucket), eps in buckets.items():
            if len(eps) < 2:
                continue
            sorted_eps = sorted(eps, key=lambda e: e.total_reward)
            worst = sorted_eps[0]
            best = sorted_eps[-1]
            if best.total_reward == worst.total_reward:
                continue
            record = {
                "chosen": {
                    "episode_id": best.id,
                    "task": task,
                    "total_reward": best.total_reward,
                    "passed": best.passed,
                },
                "rejected": {
                    "episode_id": worst.id,
                    "task": task,
                    "total_reward": worst.total_reward,
                    "passed": worst.passed,
                },
            }
            f.write(json.dumps(record) + "\n")


def _write_grpo_rollouts(env_name: str, db: Session, out_dir: Path) -> None:
    episodes = _get_episodes(env_name, db)
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


_FORMAT_WRITERS = {
    "trajectories": _write_trajectories,
    "rewards": _write_rewards,
    "verifier_results": _write_verifier_results,
    "sft_pairs": _write_sft_pairs,
    "preference_pairs": _write_preference_pairs,
    "grpo_rollouts": _write_grpo_rollouts,
}


def run_export(export_job_id: str, db: Session) -> None:
    job = db.get(ExportJob, export_job_id)
    if job is None:
        raise ValueError(f"ExportJob {export_job_id} not found")

    job.status = "running"
    db.commit()

    try:
        formats: list[str] = json.loads(job.formats)
        out_dir = BASE_DIR / job.env_name / "exports" / export_job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            writer = _FORMAT_WRITERS.get(fmt)
            if writer is None:
                logger.warning("Unknown export format: %s", fmt)
                continue
            writer(job.env_name, db, out_dir)

        job.output_path = str(out_dir)
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as exc:
        logger.exception("ExportJob %s failed: %s", export_job_id, exc)
        job.status = "failed"
        job.error = str(exc)
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        raise

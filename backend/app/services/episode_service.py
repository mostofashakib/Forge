# backend/app/services/episode_service.py
from __future__ import annotations
import json
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from backend.app.models import Episode, EpisodeStep
from forge.runtime.replay import EpisodeRecord, ReplayService
from forge.runtime.clustering import FailureClusterer


def create_episode(
    episode_id: str,
    env_name: str,
    task_name: str,
    seed: int,
    agent_id: str,
    db: Session,
    jsonl_path: str | None = None,
) -> Episode:
    ep = Episode(
        id=episode_id,
        env_name=env_name,
        task_name=task_name,
        seed=seed,
        agent_id=agent_id,
        status="running",
        total_steps=0,
        total_reward=0.0,
        passed=False,
        started_at=datetime.now(timezone.utc),
        jsonl_path=jsonl_path,
    )
    db.add(ep)
    db.commit()
    return ep


def get_episode(episode_id: str, db: Session) -> Episode | None:
    return db.get(Episode, episode_id)


def get_episode_steps(episode_id: str, db: Session) -> list[EpisodeStep]:
    return (
        db.query(EpisodeStep)
        .filter_by(episode_id=episode_id)
        .order_by(EpisodeStep.step_index)
        .all()
    )


def list_episodes(env_name: str, db: Session, limit: int = 20) -> list[Episode]:
    return (
        db.query(Episode)
        .filter_by(env_name=env_name)
        .order_by(Episode.started_at.desc())
        .limit(limit)
        .all()
    )


def get_stats(env_name: str, db: Session) -> dict:
    episodes = (
        db.query(Episode)
        .filter_by(env_name=env_name, status="completed")
        .order_by(Episode.started_at.desc())
        .limit(100)
        .all()
    )
    n = len(episodes)
    if n == 0:
        return {
            "pass_rate": 0.0,
            "avg_reward": 0.0,
            "avg_steps": 0.0,
            "policy_violation_count": 0,
            "top_failures": [],
        }

    pass_rate = sum(1 for ep in episodes if ep.passed) / n
    avg_reward = sum(ep.total_reward for ep in episodes) / n
    avg_steps = sum(ep.total_steps for ep in episodes) / n

    episode_ids = [ep.id for ep in episodes]
    all_steps = (
        db.query(EpisodeStep)
        .filter(EpisodeStep.episode_id.in_(episode_ids))
        .all()
    )
    violation_episode_ids: set[str] = set()
    for step in all_steps:
        try:
            events = json.loads(step.events)
        except (json.JSONDecodeError, TypeError):
            continue
        if any(e.get("type") == "policy_violation" for e in events):
            violation_episode_ids.add(step.episode_id)
    policy_violation_count = len(violation_episode_ids)

    failed_episodes = [ep for ep in episodes if not ep.passed]
    replay = ReplayService()
    records = [replay.load_episode(ep.id, db) for ep in failed_episodes]
    clusters = FailureClusterer().cluster(records)

    return {
        "pass_rate": round(pass_rate, 4),
        "avg_reward": round(avg_reward, 4),
        "avg_steps": round(avg_steps, 4),
        "policy_violation_count": policy_violation_count,
        "top_failures": [
            {
                "check_name": c.check_name,
                "count": c.count,
                "episode_ids": c.episode_ids,
            }
            for c in clusters
        ],
    }

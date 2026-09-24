from __future__ import annotations
import json
from collections import defaultdict
from sqlalchemy.orm import Session, defer
from backend.app.models import Episode, EpisodeStep
from forge.runtime.replay import EpisodeRecord
from forge.runtime.clustering import FailureClusterer


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
        .options(defer(EpisodeStep.diff))
        .order_by(EpisodeStep.episode_id, EpisodeStep.step_index)
        .all()
    )
    steps_by_episode: dict[str, list[EpisodeStep]] = defaultdict(list)
    violation_episode_ids: set[str] = set()
    for step in all_steps:
        steps_by_episode[step.episode_id].append(step)
        try:
            events = json.loads(step.events)
        except (json.JSONDecodeError, TypeError):
            continue
        if any(e.get("type") == "policy_violation" for e in events):
            violation_episode_ids.add(step.episode_id)
    policy_violation_count = len(violation_episode_ids)

    records = [
        EpisodeRecord(episode=ep, steps=steps_by_episode[ep.id])
        for ep in episodes
        if not ep.passed
    ]
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

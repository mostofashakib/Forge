from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from sqlalchemy.orm import Session
from ._queries import get_episodes


def write(env_name: str, db: Session, out_dir: Path) -> None:
    episodes = get_episodes(env_name, db)
    buckets: dict[tuple[str, int], list] = defaultdict(list)
    for ep in episodes:
        key = (ep.task_name, ep.seed // 10)
        buckets[key].append(ep)

    with (out_dir / "preference_pairs.jsonl").open("w") as f:
        for (task, _bucket), eps in buckets.items():
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

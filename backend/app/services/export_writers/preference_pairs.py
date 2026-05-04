from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from sqlalchemy.orm import Session
from backend.app.models import Episode, EpisodeStep
from ._queries import get_episodes, get_steps


def _action_to_command(action_raw: str) -> str:
    try:
        action = json.loads(action_raw) if action_raw else {}
    except (json.JSONDecodeError, TypeError):
        return str(action_raw)
    return action.get("command") or action.get("cmd") or json.dumps(action)


def _episode_to_messages(ep: Episode, steps: list[EpisodeStep]) -> list[dict]:
    commands = [_action_to_command(s.action) for s in steps]
    return [
        {
            "role": "user",
            "content": f"Task: {ep.task_name}\nEnvironment: {ep.env_name}",
        },
        {
            "role": "assistant",
            "content": "\n".join(f"$ {c}" for c in commands),
        },
    ]


def write(env_name: str, db: Session, out_dir: Path) -> None:
    """Preference pairs (DPO) — chosen/rejected trajectory pairs ranked by total reward.

    Episodes are bucketed by (task_name, seed // 10) so comparisons are
    between runs on the same task under similar conditions.
    Compatible with TRL DPOTrainer and LlamaFactory.
    """
    episodes = get_episodes(env_name, db)
    buckets: dict[tuple[str, int], list[Episode]] = defaultdict(list)
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

            best_steps = get_steps(best.id, db)
            worst_steps = get_steps(worst.id, db)

            record = {
                "chosen": _episode_to_messages(best, best_steps),
                "rejected": _episode_to_messages(worst, worst_steps),
                "chosen_reward": best.total_reward,
                "rejected_reward": worst.total_reward,
                "task": task,
                "chosen_passed": best.passed,
                "rejected_passed": worst.passed,
            }
            f.write(json.dumps(record) + "\n")

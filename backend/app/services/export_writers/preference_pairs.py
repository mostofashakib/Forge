from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from sqlalchemy.orm import Session

from backend.app.models import Episode, EpisodeStep
from forge.contracts.dataset_rows import PreferencePair, actions_text, exchange

from ._queries import get_episodes, get_steps_by_episode
from .files import write_jsonl
from .gate import DatasetGate


def write(env_name: str, db: Session, out_dir: Path, gate: DatasetGate) -> None:
    """Preference pairs (DPO) — chosen/rejected trajectory pairs ranked by total reward.

    Episodes are bucketed by (task_name, seed // 10) so comparisons are
    between runs on the same task under similar conditions.
    Compatible with TRL DPOTrainer and LlamaFactory.
    """
    buckets: dict[tuple[str, int], list[Episode]] = defaultdict(list)
    for ep in get_episodes(env_name, db):
        buckets[(ep.task_name, ep.seed // 10)].append(ep)
    steps_by_episode = get_steps_by_episode(env_name, db)
    pairs = (
        pair for eps in buckets.values()
        if (pair := _pair(eps, steps_by_episode)) is not None
    )
    write_jsonl(out_dir / "preference_pairs.jsonl", gate.admitted(pairs, "preference_pairs"))


def _pair(eps: list[Episode], steps_by_episode: dict[str, list[EpisodeStep]]) -> dict | None:
    """The best and worst episode of a bucket, or None when they tie or the bucket has one episode."""
    if len(eps) < 2:
        return None
    ranked = sorted(eps, key=lambda e: e.total_reward)
    worst, best = ranked[0], ranked[-1]
    if best.total_reward == worst.total_reward:
        return None
    return PreferencePair(
        env_name=best.env_name,
        task=best.task_name,
        chosen=_messages(best, steps_by_episode.get(best.id, [])),
        rejected=_messages(worst, steps_by_episode.get(worst.id, [])),
        chosen_reward=best.total_reward,
        rejected_reward=worst.total_reward,
        chosen_passed=best.passed,
        rejected_passed=worst.passed,
    ).model_dump()


def _messages(ep: Episode, steps: list[EpisodeStep]) -> list[dict]:
    return exchange(f"Task: {ep.task_name}\nEnvironment: {ep.env_name}", actions_text([s.action for s in steps]))

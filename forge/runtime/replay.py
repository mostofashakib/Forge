# forge/runtime/replay.py
from __future__ import annotations
import json
from dataclasses import dataclass, field
from sqlalchemy.orm import Session
from backend.app.models import Episode, EpisodeStep
from forge.runtime.errors import EpisodeNotFoundError


@dataclass
class ReplayMismatch:
    step_index: int
    field: str
    expected: object
    actual: object


@dataclass
class ReplayResult:
    matched: bool
    steps_replayed: int
    total_reward: float
    mismatches: list[ReplayMismatch] = field(default_factory=list)


def _recorded_field(step, name: str):
    if isinstance(step, dict):
        return step[name]
    return getattr(step, name)


def replay_episode(env, seed: int, steps) -> ReplayResult:
    """Re-execute a recorded episode and verify it reproduces exactly.

    Resets `env` with the recorded seed, replays each recorded action, and
    compares the resulting state hash and reward of every step against the
    recording. Accepts StepSnapshot objects, DB rows, or JSONL dicts — anything
    carrying `action`, `state_hash_after`, and `reward` per step.
    """
    env.reset(seed=seed)
    mismatches: list[ReplayMismatch] = []
    rewards: list[float] = []
    total_reward = 0.0
    steps = list(steps)

    for recorded in steps:
        action = _recorded_field(recorded, "action")
        if isinstance(action, str):
            action = json.loads(action)
        _obs, reward, _terminated, _truncated, _info = env.step(action)
        rewards.append(reward)
        total_reward += reward

    # One trajectory read after the loop — to_trajectory() copies the step
    # list, so reading it per step would be quadratic.
    snapshots = env.current_trajectory().steps
    for index, recorded in enumerate(steps):
        expected_hash = _recorded_field(recorded, "state_hash_after")
        if snapshots[index].state_hash_after != expected_hash:
            mismatches.append(
                ReplayMismatch(index, "state_hash_after", expected_hash, snapshots[index].state_hash_after)
            )
        expected_reward = _recorded_field(recorded, "reward")
        if rewards[index] != expected_reward:
            mismatches.append(ReplayMismatch(index, "reward", expected_reward, rewards[index]))

    return ReplayResult(
        matched=not mismatches,
        steps_replayed=len(steps),
        total_reward=total_reward,
        mismatches=mismatches,
    )


@dataclass
class EpisodeRecord:
    episode: Episode
    steps: list[EpisodeStep]


class ReplayService:
    def load_episode(self, episode_id: str, db: Session) -> EpisodeRecord:
        ep = db.get(Episode, episode_id)
        if ep is None:
            raise EpisodeNotFoundError(f"Episode {episode_id!r} not found")
        steps = (
            db.query(EpisodeStep)
            .filter_by(episode_id=episode_id)
            .order_by(EpisodeStep.step_index)
            .all()
        )
        return EpisodeRecord(episode=ep, steps=steps)

    def branch_from(self, episode_id: str, step_n: int, db: Session) -> list[dict]:
        if db.get(Episode, episode_id) is None:
            raise EpisodeNotFoundError(f"Episode {episode_id!r} not found")
        steps = (
            db.query(EpisodeStep)
            .filter(
                EpisodeStep.episode_id == episode_id,
                EpisodeStep.step_index < step_n,
            )
            .order_by(EpisodeStep.step_index)
            .all()
        )
        return [json.loads(s.action) for s in steps]

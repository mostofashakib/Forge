"""Map the verifiers' graded rewards onto a training signal.

This is the reproducible bridge between how a trajectory was *graded* and how it
is *learned from*:

- **GRPO** — group rollouts that share a prompt and standardize their episode
  rewards into group-relative advantages ``(r - mean) / (std + eps)``. A group
  whose rollouts all scored the same (or a lone rollout) has no relative signal
  and contributes nothing, so an all-equal graded set produces no update.
- **DPO** — a preference pair is a usable label only when the chosen trajectory
  was graded strictly higher than the rejected one; ties and inversions are
  dropped.

The mapping is a pure function of the exported rewards, so the training signal is
deterministic and matches the grades the rollouts were exported with.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev

from forge.training.dataset import PreferenceRecord, RolloutRecord

_EPS = 1e-6


@dataclass
class GRPOExample:
    prompt: str
    completion: str
    reward: float
    advantage: float


@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str


def grpo_advantages(rollouts: list[RolloutRecord], eps: float = _EPS) -> list[GRPOExample]:
    """Group-relative advantages, one example per rollout in a signal-bearing group."""
    groups: dict[str, list[RolloutRecord]] = {}
    for r in rollouts:
        groups.setdefault(r.prompt, []).append(r)

    examples: list[GRPOExample] = []
    for prompt, group in groups.items():
        if len(group) < 2:
            continue  # no group to be relative to
        rewards = [r.total_reward for r in group]
        spread = pstdev(rewards)
        if spread <= 0:
            continue  # every rollout scored the same — nothing to learn
        mean = sum(rewards) / len(rewards)
        for r in group:
            examples.append(GRPOExample(
                prompt=prompt,
                completion=r.completion,
                reward=r.total_reward,
                advantage=(r.total_reward - mean) / (spread + eps),
            ))
    return examples


def dpo_examples(preferences: list[PreferenceRecord]) -> list[DPOExample]:
    """Preference examples where the chosen trajectory was graded strictly higher."""
    return [
        DPOExample(prompt=p.prompt, chosen=p.chosen, rejected=p.rejected)
        for p in preferences
        if p.chosen_reward > p.rejected_reward
    ]

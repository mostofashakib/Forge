# forge/runtime/policy.py
from __future__ import annotations
import random
from typing import Callable


def seeded_random_policy(seed: int | None) -> Callable[[dict, frozenset], dict]:
    """Random action selection, deterministic when a seed is supplied.

    Passing ``None`` is reserved for the determinism-off experiment.
    """
    rng = random.Random(seed)

    def policy(obs: dict, action_types: frozenset) -> dict:
        domain_actions = sorted(action_types - {"submit"})
        return {"type": rng.choice(domain_actions or sorted(action_types))}

    return policy


class RandomPolicy:
    def __init__(
        self, action_types: frozenset[str] | set[str], seed: int | None = None
    ) -> None:
        if not action_types:
            raise ValueError("action_types must not be empty")
        self._action_types = sorted(action_types)
        self._rng = random.Random(seed)

    def act(self, obs: dict) -> dict:
        return {"type": self._rng.choice(self._action_types)}

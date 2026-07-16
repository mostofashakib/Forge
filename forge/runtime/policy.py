# forge/runtime/policy.py
from __future__ import annotations
import random
from typing import Callable


def seeded_random_policy(seed: int) -> Callable[[dict, frozenset], dict]:
    """Deterministic random action selection: same seed, same action sequence.

    The single implementation behind determinism checks, parallel rollouts,
    and CLI episode drivers.
    """
    rng = random.Random(seed)

    def policy(obs: dict, action_types: frozenset) -> dict:
        return {"type": rng.choice(sorted(action_types))}

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

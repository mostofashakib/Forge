# forge/runtime/policy.py
from __future__ import annotations
import random


class RandomPolicy:
    def __init__(self, action_types: frozenset[str] | set[str]) -> None:
        if not action_types:
            raise ValueError("action_types must not be empty")
        self._action_types = sorted(action_types)

    def act(self, obs: dict) -> dict:
        return {"type": random.choice(self._action_types)}

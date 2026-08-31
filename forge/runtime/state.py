from __future__ import annotations

import copy
import hashlib
import json

from forge.contracts import StateManager


class InProcessStateManager(StateManager):
    """State held in memory, hashed with sorted keys for cross-run stability."""

    def __init__(self, initial_state: dict) -> None:
        self._state = copy.deepcopy(initial_state)

    def get(self) -> dict:
        return copy.deepcopy(self._state)

    def apply(self, new_state: dict) -> None:
        self._state = copy.deepcopy(new_state)

    def hash(self) -> str:
        serialized = json.dumps(self._state, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        return f"sha256:{digest}"


# The pre-contracts name. Retained so existing imports and generated packages
# keep working.
StateStore = InProcessStateManager

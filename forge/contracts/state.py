"""How state is tracked across turns."""
from __future__ import annotations

from abc import ABC, abstractmethod


class StateManager(ABC):
    """Owns the environment's state and its content-addressed hash.

    `snapshot`/`restore` are concrete because only the container family
    supports named slots today (POST /forge/snapshot, POST /forge/restore/{slot}).
    An implementation that does not support them inherits a loud failure rather
    than a silent no-op.
    """

    @abstractmethod
    def get(self) -> dict:
        """Current state. Implementations return a copy, never a live reference."""

    @abstractmethod
    def apply(self, state: dict) -> None:
        """Replace the current state."""

    @abstractmethod
    def hash(self) -> str:
        """Stable content hash of the current state, prefixed with its algorithm."""

    def snapshot(self, slot: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support named state slots"
        )

    def restore(self, slot: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support named state slots"
        )

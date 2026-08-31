"""What the model sees back after an action."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from forge.contracts.types import Observation

if TYPE_CHECKING:
    from forge.runtime.context import RuntimeContext


class ObservationEncoder(ABC):
    """Turns raw environment state into what the agent is allowed to see.

    This is the seam where redaction and role-based filtering belong: the
    encoder decides what leaves the environment, so a filter cannot be bypassed
    by reading state directly.
    """

    @abstractmethod
    def encode(self, state: dict, ctx: "RuntimeContext") -> Observation:
        """Render state as an Observation."""

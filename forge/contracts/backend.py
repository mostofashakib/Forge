"""Where actions actually run."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from forge.contracts.types import Action, ActionResult

if TYPE_CHECKING:
    from forge.runtime.context import RuntimeContext
    from forge.runtime.transition import TransitionResult


class TransitionHandler(ABC):
    """One action's state transition, in-process.

    Replaces the bare Callable the transition registry accepted. A handler with
    the wrong signature is now rejected when it is registered rather than when
    it is first invoked, mid-episode.
    """

    @abstractmethod
    def apply(
        self, state: dict, action: Action, ctx: "RuntimeContext"
    ) -> "TransitionResult":
        """Return the new state and the events this action emitted."""


class ExecutionBackend(ABC):
    """Executes one action wherever the environment actually lives.

    In-process, a container over HTTP, a shell over docker exec, or a browser
    over CDP. `close` is concrete because a stateless backend holds nothing to
    release.
    """

    @abstractmethod
    def execute(
        self, action: Action, state: dict, ctx: "RuntimeContext"
    ) -> ActionResult:
        """Run the action and return the resulting state and events."""

    def close(self) -> None:
        """Release any held resource. No-op by default."""
        return None

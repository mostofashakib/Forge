"""How an episode ends."""
from __future__ import annotations

from abc import ABC, abstractmethod

from forge.contracts.types import StepOutcome, Termination


class TerminationPolicy(ABC):
    """Decides, after each step, whether the episode is over.

    Returning None means continue. Policies are consulted in order by the
    controller, so each one answers only about its own stopping condition.
    """

    @abstractmethod
    def check(self, outcome: StepOutcome) -> Termination | None:
        """Return a Termination to stop, or None to continue."""

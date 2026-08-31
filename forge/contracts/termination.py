"""How an episode ends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from forge.contracts.types import StepOutcome, Termination

if TYPE_CHECKING:
    from forge.contracts.episode import BaseEpisodeConfig


class TerminationPolicy(ABC):
    """Decides, after each step, whether the episode is over.

    Returning None means continue. Policies are consulted in order by the
    controller, so each one answers only about its own stopping condition.
    """

    @abstractmethod
    def check(self, outcome: StepOutcome) -> Termination | None:
        """Return a Termination to stop, or None to continue."""


class ThresholdTerminationPolicy(TerminationPolicy):
    """Success / dead-end / divergence, in that priority order.

    The priority matters: a high score on an unchanged state is a success, not
    a dead end. `observe` is the pre-contracts API the three runners call and is
    kept as a thin wrapper.
    """

    def __init__(self, config: "BaseEpisodeConfig") -> None:
        self._cfg = config
        self._markers: list[object] = []
        self._below_threshold_count = 0

    def check(self, outcome: StepOutcome) -> Termination | None:
        reason = self.observe(outcome.score, outcome.state_hash)
        return Termination(reason=reason) if reason else None

    def observe(self, score: float, marker: object = None) -> str | None:
        self._markers.append(marker if marker is not None else round(score, 2))

        if score >= self._cfg.success_threshold:
            return "success"

        if len(self._markers) >= self._cfg.dead_end_patience:
            recent = self._markers[-self._cfg.dead_end_patience:]
            if len(set(recent)) == 1:
                return "dead_end"

        if score < self._cfg.divergence_threshold:
            self._below_threshold_count += 1
        else:
            self._below_threshold_count = 0
        if self._below_threshold_count >= self._cfg.consecutive_below_threshold:
            return "diverged"
        return None


class MaxStepsTerminationPolicy(TerminationPolicy):
    """The step budget, made explicit.

    Each runner previously inlined this as `step_index == max_steps - 1`.
    """

    def __init__(self, max_steps: int) -> None:
        self._max_steps = max_steps

    def check(self, outcome: StepOutcome) -> Termination | None:
        if outcome.step_index >= self._max_steps - 1:
            return Termination(reason="max_steps", truncated=True)
        return None


# The pre-contracts name, kept so the runners and their tests keep working.
TerminationMonitor = ThresholdTerminationPolicy

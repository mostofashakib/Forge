from __future__ import annotations

from typing import Protocol

from forge.runtime.snapshot import StepSnapshot


class TelemetrySink(Protocol):
    """Data-collection boundary used by an environment run.

    Runtime code emits immutable episode facts through this contract. Storage,
    database models, files, and export concerns belong to backend collectors.
    """

    def record_step(self, snapshot: StepSnapshot) -> None: ...

    def complete_episode(
        self, total_reward: float, passed: bool, total_steps: int
    ) -> None: ...

    def record_policy_violation(
        self,
        step_index: int,
        action_type: str,
        violations: list,
    ) -> None: ...


class NullTelemetrySink:
    """Explicit no-op collector for runs where persistence is disabled."""

    def record_step(self, snapshot: StepSnapshot) -> None:
        del snapshot

    def complete_episode(
        self, total_reward: float, passed: bool, total_steps: int
    ) -> None:
        del total_reward, passed, total_steps

    def record_policy_violation(
        self,
        step_index: int,
        action_type: str,
        violations: list,
    ) -> None:
        del step_index, action_type, violations

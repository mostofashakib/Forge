"""How the factory executes a task in one environment type.

A runner opens fresh, seeded sessions. `batch()` wraps a validation round
for set-up and clean-up that should happen once per round, such as a CLI
snapshot or saving an app's state so it can be put back afterwards.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from forge.taskfactory.schemas import Check, GoldenStep, TaskDraft
from forge.taskfactory.state_checks import CheckOutcome


class SeedError(RuntimeError):
    """The task's seed could not be applied, so the session never started."""


@dataclass(frozen=True)
class StepResult:
    ok: bool
    error: str = ""
    output: str = ""


class TaskSession(Protocol):
    def step(self, step: GoldenStep) -> StepResult: ...

    def evaluate(self, checks: Sequence[Check]) -> list[CheckOutcome]: ...

    def fingerprint(self) -> str: ...


class TaskRunner(Protocol):
    def batch(self) -> AbstractContextManager[None]: ...

    def session(self, draft: TaskDraft) -> AbstractContextManager[TaskSession]:
        """A fresh environment with the draft's seed applied. Raises SeedError."""
        ...


def fingerprint_of(value: Any) -> str:
    """A stable hash of JSON-shaped data."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()

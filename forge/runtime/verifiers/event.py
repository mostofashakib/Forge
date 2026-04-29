from __future__ import annotations
from forge.runtime.verification import CheckResult


class EventVerifier:
    def __init__(self, event_type: str) -> None:
        self._event_type = event_type

    def check(self, state: dict, trajectory, task: dict) -> CheckResult:
        event_types = [e.get("type") for e in trajectory.events]
        passed = self._event_type in event_types
        return CheckResult(
            name=self._event_type,
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence=None if passed else f"Event '{self._event_type}' not found in trajectory",
        )

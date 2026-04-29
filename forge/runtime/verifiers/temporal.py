from __future__ import annotations
from forge.runtime.verification import CheckResult


class TemporalVerifier:
    def __init__(self, expression: str) -> None:
        parts = expression.split(" before ", 1)
        if len(parts) != 2:
            raise ValueError(
                f"TemporalVerifier expression must be 'A before B', got: {expression!r}"
            )
        self._first = parts[0].strip()
        self._second = parts[1].strip()
        self._expression = expression

    def check(self, state: dict, trajectory, task: dict) -> CheckResult:
        event_types = [e.get("type") for e in trajectory.events]
        try:
            idx_first = event_types.index(self._first)
        except ValueError:
            return CheckResult(
                name=self._expression,
                passed=False,
                score=0.0,
                evidence=f"Event '{self._first}' not found in trajectory",
            )
        try:
            idx_second = event_types.index(self._second)
        except ValueError:
            return CheckResult(
                name=self._expression,
                passed=False,
                score=0.0,
                evidence=f"Event '{self._second}' not found in trajectory",
            )
        passed = idx_first < idx_second
        return CheckResult(
            name=self._expression,
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence=None if passed else (
                f"'{self._second}' (index {idx_second}) occurred before "
                f"'{self._first}' (index {idx_first})"
            ),
        )

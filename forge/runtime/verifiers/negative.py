from __future__ import annotations
from forge.runtime.verification import CheckResult


class NegativeVerifier:
    def __init__(self, prohibited_action: str) -> None:
        self._prohibited_action = prohibited_action

    def check(self, state: dict, trajectory, task: dict) -> CheckResult:
        for i, step in enumerate(trajectory.steps):
            if step.action.get("type") == self._prohibited_action:
                return CheckResult(
                    name=self._prohibited_action,
                    passed=False,
                    score=0.0,
                    evidence=f"Prohibited action '{self._prohibited_action}' found at step {i}",
                )
        return CheckResult(
            name=self._prohibited_action,
            passed=True,
            score=1.0,
            evidence=None,
        )

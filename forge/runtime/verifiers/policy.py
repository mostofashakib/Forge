from __future__ import annotations
from forge.runtime.verification import CheckResult


class PolicyVerifier:
    def __init__(self, forbidden_action: str) -> None:
        self._forbidden_action = forbidden_action

    def check(self, state: dict, trajectory, task: dict) -> CheckResult:
        for i, step in enumerate(trajectory.steps):
            if step.action.get("type") == self._forbidden_action:
                return CheckResult(
                    name=self._forbidden_action,
                    passed=False,
                    score=0.0,
                    evidence=f"Forbidden action '{self._forbidden_action}' found at step {i}",
                )
        return CheckResult(
            name=self._forbidden_action,
            passed=True,
            score=1.0,
            evidence=None,
        )

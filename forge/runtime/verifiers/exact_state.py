from __future__ import annotations
from forge.runtime.verification import CheckResult
from forge.runtime.safe_expression import evaluate_expression


class ExactStateVerifier:
    def __init__(self, expression: str) -> None:
        self._expression = expression

    def check(self, state: dict, trajectory, task: dict) -> CheckResult:
        try:
            result = evaluate_expression(self._expression, state)
            passed = bool(result)
        except Exception as exc:
            return CheckResult(
                name=self._expression,
                passed=False,
                score=0.0,
                evidence=f"Eval error: {exc}",
            )
        return CheckResult(
            name=self._expression,
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence=None if passed else f"Expression evaluated to {result!r} with state {state!r}",
        )

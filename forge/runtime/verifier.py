from __future__ import annotations

import functools
from typing import Callable

from forge.contracts import Verifier
from forge.runtime._signature import require_arity
from forge.runtime.verification import VerificationResult


class FunctionVerifier(Verifier):
    """Adapts a plain `(state, trajectory, task) -> VerificationResult` callable."""

    def __init__(self, fn: Callable) -> None:
        require_arity(fn, "FunctionVerifier", ("state", "trajectory", "task"))
        self._fn = fn
        functools.update_wrapper(self, fn, updated=())

    @property
    def fn(self) -> Callable:
        """The plain callable this adapter wraps."""
        return self._fn

    def verify(self, state: dict, trajectory, task) -> VerificationResult:
        return self._fn(state, trajectory, task)


class VerifierEngine:
    def __init__(self) -> None:
        self._verifiers: dict[str, Verifier] = {}

    def register(self, verifier_id: str, verifier: Verifier) -> None:
        if not isinstance(verifier, Verifier):
            raise TypeError(
                f"Verifier {verifier_id!r} must be a Verifier, got "
                f"{type(verifier).__name__}. Wrap a plain function in FunctionVerifier."
            )
        self._verifiers[verifier_id] = verifier

    def run_all(
        self, state: dict, trajectory, task: dict | None
    ) -> list[VerificationResult]:
        if task is None:
            return []
        verifier_id = task.get("verifier_id")
        if not verifier_id or verifier_id not in self._verifiers:
            return []
        return [self._verifiers[verifier_id].verify(state, trajectory, task)]

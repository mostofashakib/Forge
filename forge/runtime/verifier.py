from __future__ import annotations
from typing import Callable
from forge.runtime.verification import VerificationResult


class VerifierEngine:
    def __init__(self) -> None:
        self._verifiers: dict[str, Callable] = {}

    def register(self, verifier_id: str, fn: Callable) -> None:
        self._verifiers[verifier_id] = fn

    def run_all(
        self, state: dict, trajectory, task: dict | None
    ) -> list[VerificationResult]:
        if task is None:
            return []
        verifier_id = task.get("verifier_id")
        if not verifier_id or verifier_id not in self._verifiers:
            return []
        return [self._verifiers[verifier_id](state, trajectory, task)]

from __future__ import annotations
from pydantic import BaseModel


class CheckResult(BaseModel):
    name: str
    passed: bool
    score: float
    evidence: str | None = None


class VerificationResult(BaseModel):
    verifier_id: str
    passed: bool
    score: float
    checks: list[CheckResult]
    explanation: str = ""

    @classmethod
    def from_checks(cls, verifier_id: str, checks: list[CheckResult]) -> "VerificationResult":
        passed = all(c.passed for c in checks)
        score = sum(c.score for c in checks) / len(checks) if checks else 0.0
        return cls(verifier_id=verifier_id, passed=passed, score=score, checks=checks)

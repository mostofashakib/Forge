"""Contracts for Isolated Verifier Trust Boundary and Domain Group Adapters.

Replaces brittle, one-off per-task verifiers with a hardened Trust Boundary
and standardized group adapters (CUA, TU, Code, Data).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from forge.contracts.types import CheckResult, Task, VerificationResult


class VerifierGroup(str, Enum):
    """Supported evaluation archetype groups."""

    CUA = "cua"      # Computer-Use / UI / Browser / Desktop agents
    TU = "tu"        # Tool-Use / OpenAPI / Function-calling agents
    CODE = "code"    # Code Generation / CLI / Terminal / Git agents
    DATA = "data"    # Data / Database / SQL / Analytics agents
    CUSTOM = "custom"


class TrustLevel(str, Enum):
    """Security and integrity classification of a verification verdict."""

    VERIFIED = "verified"      # Structural & cryptographic integrity validated; criteria passed
    UNVERIFIED = "unverified"  # Standard failure without evidence of intentional tampering
    TAMPERED = "tampered"      # Anti-tampering violated: state hash drift, non-monotonic steps, spoofed events
    REJECTED = "rejected"      # Policy violation or safety boundary breach detected


class GroupVerificationSpec(BaseModel):
    """Declarative specification for domain group evaluation rules.

    Eliminates ad-hoc task verification functions by standardizing checks
    per agent archetype.
    """

    group: VerifierGroup = VerifierGroup.CUSTOM
    rules: dict[str, Any] = Field(default_factory=dict)
    milestones: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    expected_state: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrustBoundaryReport(BaseModel):
    """Auditable report emitted across the verification trust boundary."""

    trust_level: TrustLevel
    passed: bool
    score: float
    checks: list[CheckResult] = Field(default_factory=list)
    tamper_evidence: list[str] = Field(default_factory=list)
    group: VerifierGroup
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_verification_result(self, verifier_id: str) -> VerificationResult:
        """Convert to standard VerificationResult contract."""
        return VerificationResult(
            verifier_id=verifier_id,
            passed=self.passed,
            score=self.score,
            checks=self.checks,
            explanation=f"Trust boundary verdict [{self.trust_level.value}]: "
            + ("; ".join(self.tamper_evidence) if self.tamper_evidence else "Clean verification"),
        )


@runtime_checkable
class VerifierGroupAdapter(Protocol):
    """Protocol implemented by domain-specific verifier adapters."""

    @property
    def group_type(self) -> VerifierGroup:
        """The group this adapter evaluates."""
        ...

    def verify(
        self,
        spec: GroupVerificationSpec,
        state: dict[str, Any],
        trajectory: Any,
        task: Task | dict[str, Any] | None,
    ) -> list[CheckResult]:
        """Execute domain-specific verification checks inside the trust boundary."""
        ...

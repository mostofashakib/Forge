"""Contracts for Artifact Versioning, Test Isolation, and Baseline Test Suites.

Enables cryptographic versioning of prompts, verifiers, and datasets for auditability
and debugging, while strictly isolating test suites from agent observation and execution.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from forge.contracts.types import CheckResult


class ArtifactType(str, Enum):
    """Types of version-tracked artifacts in Forge."""

    PROMPT = "prompt"
    VERIFIER = "verifier"
    DATA = "data"
    TEST_SUITE = "test_suite"


class VersionedArtifact(BaseModel):
    """An immutable, version-pinned artifact representation."""

    artifact_type: ArtifactType
    name: str
    version_hash: str = Field(description="SHA-256 content digest")
    version_tag: str = Field(default="v1.0.0", description="Human-readable version tag")
    metadata: dict[str, Any] = Field(default_factory=dict)


class VersionProvenance(BaseModel):
    """Cryptographic provenance bundle pinned to an episode execution."""

    prompt_version: str
    verifier_version: str
    data_version: str
    test_suite_version: str | None = None
    composite_digest: str = Field(description="Deterministic SHA-256 of combined version hashes")
    created_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InitialTest(BaseModel):
    """One authoritative baseline test assertion, completely isolated from agent access."""

    id: str
    name: str
    assertions: list[dict[str, Any]] = Field(default_factory=list)
    test_code: str | None = None
    expected_output: Any | None = None
    is_isolated: bool = Field(default=True, description="Strictly hidden from agent observation space")


class InitialTestSuite(BaseModel):
    """Authoritative test suite chosen during grading to evaluate ground truth."""

    suite_id: str
    tests: list[InitialTest] = Field(default_factory=list)
    version_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def execute_isolated(self, state: dict[str, Any], trajectory: Any) -> list[CheckResult]:
        """Execute test checks strictly inside the grading boundary."""
        results: list[CheckResult] = []
        for test in self.tests:
            passed = True
            evidence_parts: list[str] = []
            for assertion in test.assertions:
                key = assertion.get("key")
                expected = assertion.get("expected")
                actual = state.get(key)
                if expected is not None and actual != expected:
                    passed = False
                    evidence_parts.append(f"{key}: expected {expected!r}, got {actual!r}")

            evidence = "; ".join(evidence_parts) if evidence_parts else "Assertion satisfied"
            results.append(CheckResult(
                name=f"initial_test:{test.id}",
                passed=passed,
                score=1.0 if passed else 0.0,
                evidence=evidence,
            ))
        return results

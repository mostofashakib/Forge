"""Authoritative Isolated Grader.

Selects the versioned initial test suite, verifies ground truth inside
the isolated trust boundary, verifies zero agent test leakage, and pins
cryptographic version provenance to evaluation verdicts.
"""
from __future__ import annotations

import logging
from typing import Any

from forge.contracts.evaluation import EpisodeEvaluation
from forge.contracts.types import CheckResult, RewardBreakdown, RewardComponent, VerificationResult
from forge.contracts.versioning import InitialTestSuite, VersionProvenance
from forge.runtime.reward import RewardEngine
from forge.runtime.trust_boundary.boundary import VerifierTrustBoundary
from forge.runtime.verifier import VerifierEngine
from forge.runtime.versioning.isolation import TestIsolationBoundary
from forge.runtime.versioning.registry import ArtifactVersionRegistry

logger = logging.getLogger(__name__)


class IsolatedGrader:
    """Grader that evaluates against the initial test suite while enforcing test isolation."""

    def __init__(
        self,
        registry: ArtifactVersionRegistry | None = None,
        verifier_engine: VerifierEngine | None = None,
        reward_engine: RewardEngine | None = None,
        initial_test_suite: InitialTestSuite | None = None,
    ) -> None:
        self.registry = registry or ArtifactVersionRegistry()
        self.verifier_engine = verifier_engine or VerifierEngine()
        self.reward_engine = reward_engine or RewardEngine()
        self.initial_test_suite = initial_test_suite

    def resolve_initial_test_suite(self, task: dict[str, Any] | None) -> InitialTestSuite | None:
        """Pick the authoritative initial test suite for grading."""
        if self.initial_test_suite is not None:
            return self.initial_test_suite

        if task and "initial_test_suite" in task:
            raw_suite = task["initial_test_suite"]
            if isinstance(raw_suite, InitialTestSuite):
                return raw_suite
            if isinstance(raw_suite, dict):
                return InitialTestSuite(**raw_suite)

        # Check task metadata
        if task and "metadata" in task and "initial_test_suite" in task["metadata"]:
            raw_suite = task["metadata"]["initial_test_suite"]
            if isinstance(raw_suite, InitialTestSuite):
                return raw_suite
            if isinstance(raw_suite, dict):
                return InitialTestSuite(**raw_suite)

        return None

    def grade(
        self,
        state: dict[str, Any],
        trajectory: Any,
        task: dict[str, Any] | None,
        *,
        prompt: Any | None = None,
        invalid_action_count: int = 0,
        reason: str = "completed",
    ) -> EpisodeEvaluation:
        """Grade the completed episode, evaluating the initial test suite inside the trust boundary."""
        # 1. Audit Agent Isolation: ensure agent NEVER attempted to inspect or touch tests
        isolation_breaches = TestIsolationBoundary.audit_agent_isolation(trajectory)
        if isolation_breaches:
            logger.warning("Agent isolation breach detected: %s", isolation_breaches)
            breach_check = CheckResult(
                name="isolation:test_access_violation",
                passed=False,
                score=0.0,
                evidence="; ".join(isolation_breaches),
            )
            v_res = VerificationResult(
                verifier_id="isolated_grader",
                passed=False,
                score=0.0,
                checks=[breach_check],
                explanation="Failed due to test isolation breach: agent accessed test files",
            )
            return EpisodeEvaluation(
                passed=False,
                reward=RewardBreakdown(total_reward=0.0, components=[RewardComponent(name="penalty", value=0.0)]),
                verification_results=[v_res],
                reason=f"isolation_breach: {isolation_breaches[0]}",
            )

        # 2. Pick and Execute the Initial Test Suite (authoritative baseline test checks)
        initial_suite = self.resolve_initial_test_suite(task)
        initial_test_checks: list[CheckResult] = []
        if initial_suite:
            initial_test_checks = initial_suite.execute_isolated(state, trajectory)

        # 3. Execute Standard & Trust Boundary Verifiers
        verifier_results = self.verifier_engine.run_all(state, trajectory, task)

        # If we have initial test checks, incorporate them into verification results
        if initial_test_checks:
            suite_passed = all(c.passed for c in initial_test_checks)
            suite_score = sum(c.score for c in initial_test_checks) / len(initial_test_checks)
            initial_v_res = VerificationResult(
                verifier_id=f"initial_tests:{initial_suite.suite_id if initial_suite else 'default'}",
                passed=suite_passed,
                score=suite_score,
                checks=initial_test_checks,
                explanation="Grading evaluated against authoritative initial test suite",
            )
            verifier_results.insert(0, initial_v_res)

        # 4. Compute Authoritative Reward
        reward = self.reward_engine.compute(
            state,
            trajectory,
            verifier_results,
            {**(task or {}), "invalid_action_count": invalid_action_count},
        )

        all_passed = bool(verifier_results and all(r.passed for r in verifier_results))

        # 5. Build and Pin Version Provenance (Prompt, Verifier, Data, Test Suite)
        prompt_content = prompt or task.get("prompt") if task else "default_prompt"
        verifier_content = [r.model_dump() for r in verifier_results]
        data_content = task or state
        test_suite_content = initial_suite.model_dump() if initial_suite else None

        provenance = self.registry.create_provenance(
            prompt=prompt_content,
            verifier=verifier_content,
            data=data_content,
            test_suite=test_suite_content,
            metadata={"reason": reason},
        )

        return EpisodeEvaluation(
            passed=all_passed,
            reward=reward,
            verification_results=verifier_results,
            reason=reason,
        )

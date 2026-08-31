from __future__ import annotations

from collections.abc import Sequence

import pytest

from forge.contracts import (
    CheckResult,
    RewardBreakdown,
    RewardComponent,
    Rubric,
    Task,
    VerificationResult,
    Verifier,
)


class _AlwaysPasses(Verifier):
    def verify(self, state: dict, trajectory, task: Task | None) -> VerificationResult:
        return VerificationResult.from_checks(
            "v1", [CheckResult(name="ok", passed=True, score=1.0)]
        )


class _TaskSuccess(Rubric):
    def score(
        self,
        state: dict,
        trajectory,
        verifier_results: Sequence[VerificationResult],
        task: Task | None,
    ) -> RewardBreakdown:
        value = 1.0 if any(r.passed for r in verifier_results) else 0.0
        return RewardBreakdown(
            total_reward=value,
            components=[RewardComponent(name="task_success", value=value)],
        )


def test_a_verifier_returns_a_verification_result():
    result = _AlwaysPasses().verify({}, None, None)
    assert result.passed is True
    assert result.score == 1.0


def test_a_rubric_scores_from_verifier_results():
    verdict = _AlwaysPasses().verify({}, None, None)
    breakdown = _TaskSuccess().score({}, None, [verdict], None)
    assert breakdown.total_reward == 1.0
    assert breakdown.components[0].name == "task_success"


def test_no_verifier_results_scores_zero_not_one():
    # Negative: an unverified episode must not be rewarded by default.
    assert _TaskSuccess().score({}, None, [], None).total_reward == 0.0


def test_a_rubric_missing_score_cannot_be_instantiated():
    class Incomplete(Rubric):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_a_verifier_missing_verify_cannot_be_instantiated():
    class Incomplete(Verifier):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()

from __future__ import annotations

import pytest

from forge.contracts import StepOutcome, Termination, TerminationPolicy


class _StopAtScore(TerminationPolicy):
    def check(self, outcome: StepOutcome) -> Termination | None:
        if outcome.score >= 0.9:
            return Termination(reason="success")
        return None


def test_a_policy_returns_none_to_continue():
    assert _StopAtScore().check(StepOutcome(step_index=0, score=0.1)) is None


def test_a_policy_returns_a_termination_to_stop():
    decision = _StopAtScore().check(StepOutcome(step_index=3, score=0.95))
    assert decision is not None
    assert decision.reason == "success"
    assert decision.truncated is False


def test_a_policy_missing_check_cannot_be_instantiated():
    # Negative: an incomplete policy fails at instantiation.
    class Incomplete(TerminationPolicy):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


@pytest.mark.parametrize(
    "reason",
    ["max_steps", "max_tokens", "max_wall_clock_time", "max_cost", "step_budget_exhausted", "max_steps_reached"],
)
def test_budget_reasons_count_as_truncation(reason):
    from forge.contracts.termination import is_budget_reason

    assert is_budget_reason(reason) is True


@pytest.mark.parametrize("reason", [None, "", "submitted", "success", "dead_end", "runner_error: boom"])
def test_non_budget_reasons_do_not_count_as_truncation(reason):
    from forge.contracts.termination import is_budget_reason

    assert is_budget_reason(reason) is False

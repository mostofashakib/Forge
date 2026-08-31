"""ThresholdTerminationPolicy must reproduce TerminationMonitor exactly."""
from __future__ import annotations

from forge.contracts import BaseEpisodeConfig, StepOutcome
from forge.contracts.termination import (
    MaxStepsTerminationPolicy,
    ThresholdTerminationPolicy,
)


def _config(**kwargs) -> BaseEpisodeConfig:
    return BaseEpisodeConfig(objective="close the ticket", **kwargs)


def test_reaching_the_success_threshold_terminates():
    policy = ThresholdTerminationPolicy(_config())
    assert policy.check(StepOutcome(step_index=0, score=0.95)).reason == "success"


def test_an_unchanged_marker_for_the_patience_window_is_a_dead_end():
    policy = ThresholdTerminationPolicy(_config(dead_end_patience=3))
    outcomes = [
        policy.check(StepOutcome(step_index=i, score=0.5, state_hash="same"))
        for i in range(3)
    ]
    assert outcomes[-1].reason == "dead_end"


def test_sustained_low_scores_diverge():
    policy = ThresholdTerminationPolicy(
        _config(divergence_threshold=0.2, consecutive_below_threshold=2, dead_end_patience=99)
    )
    policy.check(StepOutcome(step_index=0, score=0.1, state_hash="a"))
    assert policy.check(StepOutcome(step_index=1, score=0.1, state_hash="b")).reason == "diverged"


def test_success_outranks_dead_end():
    # Negative: a high score on an unchanged state is a success, not a dead end.
    policy = ThresholdTerminationPolicy(_config(dead_end_patience=1))
    assert policy.check(StepOutcome(step_index=0, score=0.99, state_hash="same")).reason == "success"


def test_a_recovering_score_resets_the_divergence_counter():
    # False-positive guard: one good step must clear the streak.
    policy = ThresholdTerminationPolicy(
        _config(divergence_threshold=0.2, consecutive_below_threshold=2, dead_end_patience=99)
    )
    policy.check(StepOutcome(step_index=0, score=0.1, state_hash="a"))
    policy.check(StepOutcome(step_index=1, score=0.5, state_hash="b"))
    assert policy.check(StepOutcome(step_index=2, score=0.1, state_hash="c")) is None


def test_the_legacy_observe_api_still_works():
    # The three runners call observe(); it must survive the rename.
    policy = ThresholdTerminationPolicy(_config())
    assert policy.observe(0.95) == "success"
    assert policy.observe(0.5) is None


def test_max_steps_truncates_rather_than_terminates():
    policy = MaxStepsTerminationPolicy(max_steps=3)
    assert policy.check(StepOutcome(step_index=1)) is None
    decision = policy.check(StepOutcome(step_index=2))
    assert decision.reason == "max_steps"
    assert decision.truncated is True

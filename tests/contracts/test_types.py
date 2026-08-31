from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge.contracts import (
    Action,
    CheckResult,
    Observation,
    Task,
    Termination,
    VerificationResult,
)


def test_action_round_trips_through_a_plain_dict():
    action = Action.from_dict({"type": "close_ticket", "ticket_id": "t_1"})
    assert action.type == "close_ticket"
    assert action.params == {"ticket_id": "t_1"}
    assert action.to_dict() == {"type": "close_ticket", "ticket_id": "t_1"}


def test_action_requires_a_type():
    # Negative: an action with no type is not an action.
    with pytest.raises(KeyError):
        Action.from_dict({"ticket_id": "t_1"})


def test_observation_defaults_are_empty_not_none():
    # False-positive guard: an observation with no text still has a usable
    # payload and blocks, so consumers never branch on None.
    obs = Observation()
    assert obs.payload == {}
    assert obs.blocks == []
    assert obs.text is None


def test_verification_result_from_checks_averages_scores():
    result = VerificationResult.from_checks(
        "v1",
        [
            CheckResult(name="a", passed=True, score=1.0),
            CheckResult(name="b", passed=False, score=0.0),
        ],
    )
    assert result.passed is False
    assert result.score == 0.5


def test_verification_result_from_no_checks_does_not_pass_vacuously():
    # Negative: zero checks must not average to a passing score.
    result = VerificationResult.from_checks("v1", [])
    assert result.score == 0.0


def test_task_rejects_a_missing_objective():
    with pytest.raises(ValidationError):
        Task(id="t1")


def test_termination_defaults_to_not_truncated():
    assert Termination(reason="success").truncated is False

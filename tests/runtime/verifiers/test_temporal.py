import pytest
from dataclasses import dataclass, field
from forge.runtime.verifiers.temporal import TemporalVerifier


@dataclass
class _FakeTraj:
    steps: list = field(default_factory=list)
    events: list = field(default_factory=list)


def _traj(*event_types: str) -> _FakeTraj:
    return _FakeTraj(events=[{"type": t} for t in event_types])


def test_passes_when_a_before_b():
    v = TemporalVerifier("ask_for_id before offer_refund")
    result = v.check({}, _traj("ask_for_id", "offer_refund"), {})
    assert result.passed
    assert result.score == 1.0


def test_fails_when_b_before_a():
    v = TemporalVerifier("ask_for_id before offer_refund")
    result = v.check({}, _traj("offer_refund", "ask_for_id"), {})
    assert not result.passed
    assert "offer_refund" in result.evidence


def test_fails_when_first_event_missing():
    v = TemporalVerifier("ask_for_id before offer_refund")
    result = v.check({}, _traj("offer_refund"), {})
    assert not result.passed
    assert "ask_for_id" in result.evidence


def test_fails_when_second_event_missing():
    v = TemporalVerifier("ask_for_id before offer_refund")
    result = v.check({}, _traj("ask_for_id"), {})
    assert not result.passed
    assert "offer_refund" in result.evidence


def test_fails_on_empty_trajectory():
    v = TemporalVerifier("a before b")
    result = v.check({}, _traj(), {})
    assert not result.passed


def test_raises_on_bad_expression():
    with pytest.raises(ValueError, match="must be 'A before B'"):
        TemporalVerifier("single_event")


def test_name_is_expression():
    v = TemporalVerifier("a before b")
    result = v.check({}, _traj("a", "b"), {})
    assert result.name == "a before b"

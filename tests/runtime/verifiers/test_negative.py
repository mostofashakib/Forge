from dataclasses import dataclass, field
from forge.runtime.verifiers.negative import NegativeVerifier


@dataclass
class _FakeStep:
    action: dict = field(default_factory=dict)


@dataclass
class _FakeTraj:
    steps: list = field(default_factory=list)
    events: list = field(default_factory=list)


def test_passes_when_prohibited_action_absent():
    v = NegativeVerifier("premature_close")
    traj = _FakeTraj(steps=[_FakeStep(action={"type": "reply_to_customer"})])
    result = v.check({}, traj, {})
    assert result.passed
    assert result.score == 1.0


def test_fails_when_prohibited_action_present():
    v = NegativeVerifier("premature_close")
    traj = _FakeTraj(steps=[_FakeStep(action={"type": "premature_close"})])
    result = v.check({}, traj, {})
    assert not result.passed
    assert "premature_close" in result.evidence


def test_passes_on_empty_trajectory():
    v = NegativeVerifier("premature_close")
    result = v.check({}, _FakeTraj(), {})
    assert result.passed


def test_name_is_prohibited_action():
    v = NegativeVerifier("bad_action")
    result = v.check({}, _FakeTraj(), {})
    assert result.name == "bad_action"

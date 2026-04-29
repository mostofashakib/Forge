from dataclasses import dataclass, field
from forge.runtime.verifiers.policy import PolicyVerifier


@dataclass
class _FakeStep:
    action: dict = field(default_factory=dict)


@dataclass
class _FakeTraj:
    steps: list = field(default_factory=list)
    events: list = field(default_factory=list)


def test_passes_when_forbidden_action_absent():
    v = PolicyVerifier("call_hidden_api")
    traj = _FakeTraj(steps=[_FakeStep(action={"type": "reply_to_customer"})])
    result = v.check({}, traj, {})
    assert result.passed
    assert result.score == 1.0


def test_fails_when_forbidden_action_present():
    v = PolicyVerifier("call_hidden_api")
    traj = _FakeTraj(steps=[
        _FakeStep(action={"type": "reply_to_customer"}),
        _FakeStep(action={"type": "call_hidden_api"}),
    ])
    result = v.check({}, traj, {})
    assert not result.passed
    assert "call_hidden_api" in result.evidence
    assert "step 1" in result.evidence


def test_passes_on_empty_trajectory():
    v = PolicyVerifier("call_hidden_api")
    result = v.check({}, _FakeTraj(), {})
    assert result.passed


def test_name_is_forbidden_action():
    v = PolicyVerifier("bad_action")
    result = v.check({}, _FakeTraj(), {})
    assert result.name == "bad_action"

from dataclasses import dataclass, field
from forge.runtime.verifiers.event import EventVerifier


@dataclass
class _FakeTraj:
    steps: list = field(default_factory=list)
    events: list = field(default_factory=list)


def test_passes_when_event_present():
    v = EventVerifier("ticket_resolved")
    result = v.check({}, _FakeTraj(events=[{"type": "ticket_resolved"}]), {})
    assert result.passed
    assert result.score == 1.0


def test_fails_when_event_absent():
    v = EventVerifier("ticket_resolved")
    result = v.check({}, _FakeTraj(events=[{"type": "other_event"}]), {})
    assert not result.passed
    assert "ticket_resolved" in result.evidence


def test_passes_when_event_among_many():
    v = EventVerifier("reply_sent")
    result = v.check(
        {},
        _FakeTraj(events=[{"type": "action_taken"}, {"type": "reply_sent"}, {"type": "closed"}]),
        {},
    )
    assert result.passed


def test_fails_on_empty_events():
    v = EventVerifier("reply_sent")
    result = v.check({}, _FakeTraj(events=[]), {})
    assert not result.passed


def test_name_is_event_type():
    v = EventVerifier("my_event")
    result = v.check({}, _FakeTraj(events=[{"type": "my_event"}]), {})
    assert result.name == "my_event"

"""Persona cadence: who acts, when, and how the traits move it."""
from __future__ import annotations

import random

from forge.contracts.persona import PersonaTick
from forge.personas.scheduler import PersonaScheduleState, TraitScheduler

from tests.personas.conftest import persona


def tick(step_index: int, action_type: str | None = "page_nurse", events=None):
    return PersonaTick(
        step_index=step_index,
        agent_action={"type": action_type} if action_type else None,
        events=events or [],
    )


def scheduler(max_due: int = 1):
    state = PersonaScheduleState()
    return TraitScheduler(state, max_due=max_due), state


def test_persona_with_no_allowed_actions_is_never_due():
    """The inert default: configuration, not personality, decides this."""
    sched, _ = scheduler()
    spec = persona("nurse", allowed_actions=[], activity=100)
    assert sched.due([spec], tick(0), random.Random(0)) == []


def test_matching_wake_action_makes_a_persona_due_immediately():
    sched, _ = scheduler()
    spec = persona("nurse", wake_on=["page_nurse"], latency_steps=0)
    assert sched.due([spec], tick(0, "page_nurse"), random.Random(0)) == ["nurse"]


def test_non_matching_wake_action_leaves_a_persona_quiet():
    sched, _ = scheduler()
    spec = persona("nurse", wake_on=["page_nurse"], activity=0)
    assert sched.due([spec], tick(0, "write_note"), random.Random(0)) == []


def test_empty_wake_on_means_any_agent_action_wakes_them():
    sched, _ = scheduler()
    spec = persona("nurse", wake_on=[], latency_steps=0, activity=0)
    assert sched.due([spec], tick(0, "anything_at_all"), random.Random(0)) == ["nurse"]


def test_being_named_in_an_event_wakes_a_persona_the_action_type_would_not():
    sched, _ = scheduler()
    spec = persona("nurse", wake_on=["page_nurse"], activity=0)
    due = sched.due(
        [spec],
        tick(0, "write_note", events=[{"type": "note", "recipient": "nurse"}]),
        random.Random(0),
    )
    assert due == ["nurse"]


def test_latency_delays_the_reply_by_the_configured_number_of_steps():
    sched, _ = scheduler()
    spec = persona(
        "nurse", wake_on=["page_nurse"], latency_steps=3, responsiveness=0, activity=0
    )
    rng = random.Random(0)
    assert sched.due([spec], tick(0, "page_nurse"), rng) == []
    assert sched.due([spec], tick(1, None), rng) == []
    assert sched.due([spec], tick(2, None), rng) == []
    assert sched.due([spec], tick(3, None), rng) == ["nurse"]


def test_high_responsiveness_shortens_the_same_configured_latency():
    sched, _ = scheduler()
    eager = persona(
        "nurse", wake_on=["page_nurse"], latency_steps=3, responsiveness=100, activity=0
    )
    assert sched.due([eager], tick(0, "page_nurse"), random.Random(0)) == ["nurse"]


def test_cooldown_keeps_a_persona_quiet_after_acting():
    sched, state = scheduler()
    spec = persona("nurse", wake_on=[], latency_steps=0, cooldown_steps=3, activity=0)
    rng = random.Random(0)
    assert sched.due([spec], tick(0), rng) == ["nurse"]
    state.record_action("nurse", 0)
    assert sched.due([spec], tick(1), rng) == []
    assert sched.due([spec], tick(2), rng) == []
    assert sched.due([spec], tick(3), rng) == ["nurse"]


def test_episode_budget_stops_a_persona_permanently():
    sched, state = scheduler()
    spec = persona(
        "nurse",
        wake_on=[],
        latency_steps=0,
        cooldown_steps=0,
        max_actions_per_episode=2,
    )
    rng = random.Random(0)
    for step in range(2):
        assert sched.due([spec], tick(step), rng) == ["nurse"]
        state.record_action("nurse", step)
    assert sched.due([spec], tick(9), rng) == []


def test_zero_activity_persona_never_acts_unprompted_however_eager():
    """Configuration outranks personality: activity 0 beats initiative 100."""
    sched, _ = scheduler()
    spec = persona("nurse", wake_on=["never_happens"], activity=0, initiative=100)
    rng = random.Random(0)
    assert [sched.due([spec], tick(i, "other"), rng) for i in range(20)] == [[]] * 20


def test_high_activity_persona_does_act_unprompted():
    """False-positive guard for the test above."""
    sched, state = scheduler()
    spec = persona(
        "nurse", wake_on=["never_happens"], activity=100, cooldown_steps=0
    )
    rng = random.Random(0)
    fired = 0
    for i in range(10):
        if sched.due([spec], tick(i, "other"), rng):
            fired += 1
            state.record_action("nurse", i)
    assert fired > 0


def test_max_due_caps_how_many_personas_act_on_one_step():
    sched, _ = scheduler(max_due=2)
    roster = [persona(f"p{i}", wake_on=[], latency_steps=0) for i in range(5)]
    assert len(sched.due(roster, tick(0), random.Random(0))) == 2


def test_someone_being_waited_on_is_scheduled_before_a_volunteer():
    sched, _ = scheduler(max_due=1)
    waiting = persona("nurse", wake_on=["page_nurse"], latency_steps=0, activity=0)
    volunteer = persona("visitor", wake_on=["never"], activity=100)
    assert sched.due([volunteer, waiting], tick(0, "page_nurse"), random.Random(0)) == [
        "nurse"
    ]


def test_same_seed_produces_the_same_cadence():
    roster = [persona(f"p{i}", wake_on=["never"], activity=50, cooldown_steps=0) for i in range(4)]

    def run(seed):
        sched, state = scheduler(max_due=2)
        rng = random.Random(seed)
        out = []
        for step in range(15):
            due = sched.due(roster, tick(step, "other"), rng)
            for pid in due:
                state.record_action(pid, step)
            out.append(due)
        return out

    assert run(11) == run(11)
    assert run(11) != run(12)

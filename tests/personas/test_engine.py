"""The tick that gives simulated humans their turns."""
from __future__ import annotations

import random

from forge.contracts.persona import (
    PersonaDriver,
    PersonaPopulation,
    PersonaTurn,
    PersonaView,
)
from forge.contracts.types import Action, ActionResult
from forge.contracts.backend import ExecutionBackend
from forge.personas.engine import PersonaEngine
from forge.runtime.context import RuntimeContext

from tests.personas.conftest import POST_MESSAGE_SPEC, persona, population


class RecordingBackend(ExecutionBackend):
    """Appends a message and records the actor it was executed as."""

    def __init__(self, fail_on: str | None = None):
        self.actions: list[tuple[str, str]] = []
        self.fail_on = fail_on

    def execute(self, action: Action, state: dict, ctx) -> ActionResult:
        if self.fail_on == action.type:
            raise RuntimeError("backend rejected the action")
        self.actions.append((action.type, ctx.actor_id))
        messages = dict(state.get("messages", {}))
        key = f"m{len(messages)}"
        messages[key] = {"author": ctx.actor_id}
        return ActionResult(
            state={**state, "messages": messages},
            events=[{"type": "message_posted", "entity_id": key}],
        )


class FixedDriver(PersonaDriver):
    def __init__(self, action_type="post_message", params=None):
        self.action_type = action_type
        self.params = params or {"body": "hello"}

    def act(self, view: PersonaView) -> PersonaTurn:
        return PersonaTurn(
            persona_id=view.persona.id,
            action=Action(type=self.action_type, params=self.params),
            trigger=view.trigger,
        )


class SilentDriver(PersonaDriver):
    def act(self, view: PersonaView) -> PersonaTurn:
        return PersonaTurn(persona_id=view.persona.id)


def engine(pop, driver=None, **kwargs):
    kwargs.setdefault("environment_actions", ["post_message"])
    kwargs.setdefault("tool_specs", [POST_MESSAGE_SPEC])
    return PersonaEngine(pop, driver=driver, **kwargs)


def ctx():
    return RuntimeContext(seed=1, deterministic=True)


def tick(eng, backend, state=None, step_index=0, action_type="page_nurse"):
    return eng.tick(
        backend=backend,
        state=state if state is not None else {"messages": {}},
        ctx=ctx(),
        step_index=step_index,
        agent_action={"type": action_type},
        events=[],
    )


def test_disabled_population_never_touches_the_world():
    eng = engine(PersonaPopulation(enabled=False, roster=[persona("nurse")]))
    eng.reset(1)
    backend = RecordingBackend()
    result = tick(eng, backend, state={"messages": {}})
    assert backend.actions == []
    assert result.state == {"messages": {}}
    assert result.turns == []


def test_a_due_persona_acts_through_the_backend():
    eng = engine(population(persona("nurse", wake_on=[])), FixedDriver())
    eng.reset(1)
    backend = RecordingBackend()
    result = tick(eng, backend)
    assert backend.actions == [("post_message", "nurse")]
    assert result.acted
    assert "m0" in result.state["messages"]


def test_the_persona_is_the_actor_not_the_agent():
    eng = engine(population(persona("nurse", wake_on=[])), FixedDriver())
    eng.reset(1)
    backend = RecordingBackend()
    tick(eng, backend)
    assert backend.actions[0][1] == "nurse"


def test_the_agent_actor_id_is_restored_after_a_persona_turn():
    eng = engine(population(persona("nurse", wake_on=[])), FixedDriver())
    eng.reset(1)
    runtime_ctx = ctx()
    eng.tick(
        backend=RecordingBackend(),
        state={"messages": {}},
        ctx=runtime_ctx,
        step_index=0,
        agent_action={"type": "page_nurse"},
    )
    assert runtime_ctx.actor_id == "agent"


def test_every_turn_emits_an_observable_event_even_when_the_action_is_silent():
    class SilentBackend(ExecutionBackend):
        def execute(self, action, state, ctx):
            return ActionResult(state=state, events=[])

    eng = engine(population(persona("nurse", wake_on=[])), FixedDriver())
    eng.reset(1)
    result = tick(eng, SilentBackend())
    assert [e["type"] for e in result.events] == ["persona_turn"]
    assert result.events[0]["actor"] == "nurse"


def test_events_from_a_persona_action_are_tagged_with_who_caused_them():
    eng = engine(population(persona("nurse", wake_on=[])), FixedDriver())
    eng.reset(1)
    result = tick(eng, RecordingBackend())
    posted = [e for e in result.events if e["type"] == "message_posted"]
    assert posted[0]["actor"] == "nurse"
    assert posted[0]["actor_kind"] == "persona"


def test_an_action_outside_the_declared_space_is_blocked_not_executed():
    eng = engine(
        population(persona("nurse", wake_on=[], allowed_actions=["post_message"])),
        FixedDriver(action_type="delete_everything"),
    )
    eng.reset(1)
    backend = RecordingBackend()
    result = tick(eng, backend)
    assert backend.actions == []
    assert result.blocked
    assert "outside its action space" in result.blocked[0].blocked


def test_a_blocked_turn_is_still_recorded_in_the_transcript():
    eng = engine(
        population(persona("nurse", wake_on=[])),
        FixedDriver(action_type="delete_everything"),
    )
    eng.reset(1)
    tick(eng, RecordingBackend())
    assert eng.transcript[0].blocked


def test_a_persona_who_declines_leaves_the_world_untouched():
    eng = engine(population(persona("nurse", wake_on=[])), SilentDriver())
    eng.reset(1)
    backend = RecordingBackend()
    result = tick(eng, backend)
    assert backend.actions == []
    assert result.turns[0].skipped == "declined"


def test_a_backend_failure_skips_the_turn_rather_than_raising():
    eng = engine(
        population(persona("nurse", wake_on=[])),
        FixedDriver(),
    )
    eng.reset(1)
    result = tick(eng, RecordingBackend(fail_on="post_message"))
    assert result.turns[0].skipped.startswith("action failed")
    assert not result.acted


def test_max_actions_per_step_caps_the_cast():
    pop = population(
        *[persona(f"p{i}", wake_on=[]) for i in range(4)], max_actions_per_step=2
    )
    eng = engine(pop, FixedDriver())
    eng.reset(1)
    backend = RecordingBackend()
    tick(eng, backend)
    assert len(backend.actions) == 2


def test_two_personas_on_one_step_each_see_the_previous_one_s_write():
    pop = population(
        persona("nurse", wake_on=[]), persona("doctor", wake_on=[]), max_actions_per_step=2
    )
    eng = engine(pop, FixedDriver())
    eng.reset(1)
    result = tick(eng, RecordingBackend())
    assert len(result.state["messages"]) == 2


def test_describe_withholds_traits_from_the_agent():
    eng = engine(population(persona("nurse", wake_on=[])), FixedDriver())
    eng.reset(1)
    described = eng.describe()
    assert described == [{"id": "nurse", "name": "Nurse", "role": "charge nurse"}]


def test_reset_clears_the_transcript_between_episodes():
    eng = engine(population(persona("nurse", wake_on=[])), FixedDriver())
    eng.reset(1)
    tick(eng, RecordingBackend())
    assert eng.transcript
    eng.reset(2)
    assert eng.transcript == []


def test_reset_clears_cooldown_so_a_new_episode_starts_fresh():
    eng = engine(
        population(persona("nurse", wake_on=[], cooldown_steps=50)), FixedDriver()
    )
    eng.reset(1)
    assert tick(eng, RecordingBackend(), step_index=0).acted
    assert not tick(eng, RecordingBackend(), step_index=1).acted
    eng.reset(2)
    assert tick(eng, RecordingBackend(), step_index=0).acted


def test_same_seed_produces_the_same_persona_stream():
    pop = population(
        *[persona(f"p{i}", wake_on=["never"], activity=50, cooldown_steps=0) for i in range(4)],
        max_actions_per_step=2,
    )

    def run(seed):
        eng = engine(pop, FixedDriver())
        eng.reset(seed)
        backend = RecordingBackend()
        for step in range(12):
            tick(eng, backend, step_index=step, action_type="other")
        return backend.actions

    assert run(7) == run(7)


def test_deterministic_driver_context_replaces_a_model_driver_and_restores_it():
    model_driver = FixedDriver()
    eng = engine(population(persona("nurse", wake_on=[])), model_driver)
    eng.reset(1)
    with eng.deterministic_driver():
        eng.reset(1)
        assert eng._driver is not model_driver
    eng.reset(1)
    assert eng._driver is model_driver

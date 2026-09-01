"""What a persona decides to do, and what happens when that goes wrong."""
from __future__ import annotations

import random

import pytest

from forge.contracts.persona import PersonaView
from forge.personas.drivers import (
    AgentPersonaDriver,
    ScriptedPersonaDriver,
    make_driver,
)

from tests.personas.conftest import POST_MESSAGE_SPEC, persona


def view(spec=None, action_space=None):
    spec = spec or persona("nurse", allowed_actions=["post_message"])
    return PersonaView(
        persona=spec.profile,
        behavior=spec.behavior,
        step_index=0,
        trigger="page_nurse",
        state={"messages": {}},
        recent_events=[],
        action_space=action_space if action_space is not None else [POST_MESSAGE_SPEC],
    )


class StubAdapter:
    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.calls = []

    def act(self, obs, action_types):
        self.calls.append((obs, action_types))
        if self.raises is not None:
            raise self.raises
        return self.response


# --- scripted -------------------------------------------------------------


def test_scripted_driver_fills_required_parameters():
    turn = ScriptedPersonaDriver(random.Random(0)).act(view())
    assert turn.action.type == "post_message"
    assert "body" in turn.action.params


def test_scripted_driver_is_reproducible_under_the_same_seed():
    space = [POST_MESSAGE_SPEC, POST_MESSAGE_SPEC.model_copy(update={"name": "review_chart"})]
    first = [
        ScriptedPersonaDriver(random.Random(5)).act(view(action_space=space)).action.type
        for _ in range(1)
    ]
    second = [
        ScriptedPersonaDriver(random.Random(5)).act(view(action_space=space)).action.type
        for _ in range(1)
    ]
    assert first == second


def test_scripted_driver_declines_when_there_is_nothing_it_may_do():
    turn = ScriptedPersonaDriver().act(view(action_space=[]))
    assert turn.action is None
    assert turn.skipped == "no action available"


def test_scripted_driver_never_emits_a_float():
    """Persona actions reach state, and state rejects floats."""
    from forge.contracts.types import ToolParam, ToolSpec

    spec = ToolSpec(name="rate", params=[ToolParam(name="score", type="number")])
    turn = ScriptedPersonaDriver(random.Random(0)).act(view(action_space=[spec]))
    assert not isinstance(turn.action.params["score"], float)


def test_reset_reseeds_so_two_episodes_make_the_same_choices():
    space = [
        POST_MESSAGE_SPEC.model_copy(update={"name": f"a{i}"}) for i in range(6)
    ]
    driver = ScriptedPersonaDriver(random.Random(1))
    first = [driver.act(view(action_space=space)).action.type for _ in range(5)]
    driver.reset(random.Random(1))
    second = [driver.act(view(action_space=space)).action.type for _ in range(5)]
    assert first == second


# --- model-backed ---------------------------------------------------------


def test_agent_driver_returns_what_the_model_chose():
    adapter = StubAdapter({"type": "post_message", "body": "on my way"})
    driver = AgentPersonaDriver(adapter_factory=lambda *_: adapter)
    turn = driver.act(view())
    assert turn.action.type == "post_message"
    assert turn.action.params["body"] == "on my way"


def test_agent_driver_is_told_only_the_actions_the_persona_may_take():
    adapter = StubAdapter({"type": "post_message"})
    AgentPersonaDriver(adapter_factory=lambda *_: adapter).act(view())
    _obs, action_types = adapter.calls[0]
    assert action_types == frozenset({"post_message"})


def test_agent_driver_prompt_payload_carries_the_persona_and_the_trigger():
    adapter = StubAdapter({"type": "post_message"})
    AgentPersonaDriver(adapter_factory=lambda *_: adapter).act(view())
    obs, _ = adapter.calls[0]
    assert obs["you"]["id"] == "nurse"
    assert obs["why_you_are_acting"] == "page_nurse"
    assert obs["actions_available_to_you"] == ["post_message"]


def test_agent_driver_reuses_one_adapter_per_persona():
    built = []

    def factory(*args):
        adapter = StubAdapter({"type": "post_message"})
        built.append(adapter)
        return adapter

    driver = AgentPersonaDriver(adapter_factory=factory)
    for _ in range(3):
        driver.act(view())
    assert len(built) == 1


def test_agent_driver_builds_a_separate_adapter_per_persona():
    built = []

    def factory(*args):
        adapter = StubAdapter({"type": "post_message"})
        built.append(adapter)
        return adapter

    driver = AgentPersonaDriver(adapter_factory=factory)
    driver.act(view(persona("nurse", allowed_actions=["post_message"])))
    driver.act(view(persona("doctor", allowed_actions=["post_message"])))
    assert len(built) == 2


def test_reset_drops_adapters_so_no_persona_remembers_the_last_episode():
    built = []

    def factory(*args):
        adapter = StubAdapter({"type": "post_message"})
        built.append(adapter)
        return adapter

    driver = AgentPersonaDriver(adapter_factory=factory)
    driver.act(view())
    driver.reset(random.Random(0))
    driver.act(view())
    assert len(built) == 2


def test_a_failing_model_call_falls_back_rather_than_ending_the_episode():
    driver = AgentPersonaDriver(
        adapter_factory=lambda *_: StubAdapter(raises=RuntimeError("no api key")),
        fallback=ScriptedPersonaDriver(random.Random(0)),
    )
    turn = driver.act(view())
    assert turn.action is not None


def test_a_failing_model_call_with_no_fallback_skips_the_turn():
    driver = AgentPersonaDriver(
        adapter_factory=lambda *_: StubAdapter(raises=RuntimeError("no api key"))
    )
    turn = driver.act(view())
    assert turn.action is None
    assert "no api key" in turn.skipped


@pytest.mark.parametrize("response", [None, "not a dict", {}, {"body": "no type"}])
def test_a_malformed_model_response_skips_the_turn(response):
    driver = AgentPersonaDriver(adapter_factory=lambda *_: StubAdapter(response))
    turn = driver.act(view())
    assert turn.action is None
    assert turn.skipped


# --- selection ------------------------------------------------------------


def test_make_driver_defaults_to_scripted():
    assert isinstance(make_driver("scripted"), ScriptedPersonaDriver)
    assert isinstance(make_driver(""), ScriptedPersonaDriver)


def test_make_driver_gives_a_model_driver_a_scripted_fallback():
    driver = make_driver("anthropic:claude-sonnet-5", random.Random(0))
    assert isinstance(driver, AgentPersonaDriver)
    assert isinstance(driver._fallback, ScriptedPersonaDriver)

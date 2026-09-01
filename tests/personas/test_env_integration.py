"""Personas inside a running in-process environment."""
from __future__ import annotations

import pytest

from forge.contracts import ToolParam
from forge.contracts.persona import PersonaDriver, PersonaTurn, PersonaView
from forge.contracts.types import Action
from forge.runtime.env_builder import EnvBuilder
from forge.runtime.errors import EnvironmentBuildError

from tests.personas.conftest import (
    MessageBoardFactory,
    persona,
    population,
    post_message,
    review_chart,
)


class ReplyDriver(PersonaDriver):
    """A colleague who always answers, in their own voice."""

    def act(self, view: PersonaView) -> PersonaTurn:
        return PersonaTurn(
            persona_id=view.persona.id,
            trigger=view.trigger,
            action=Action(
                type="post_message",
                params={"body": f"{view.persona.name} here, on it"},
            ),
        )


def builder(pop=None, driver=None):
    b = (
        EnvBuilder("ward", domain="clinical", max_steps=10)
        .with_initial_state(MessageBoardFactory())
        .with_transition(
            "post_message",
            post_message,
            description="Write a note on the ward board",
            params=[ToolParam(name="body", type="string")],
        )
        .with_transition("review_chart", review_chart)
    )
    if pop is not None:
        b = b.with_personas(pop, driver=driver)
    return b


def test_an_environment_without_personas_is_unchanged():
    env = builder().build()
    obs, info = env.reset(seed=1)
    assert "personas" not in info
    obs, _, _, _, info = env.step({"type": "post_message", "body": "hi"})
    assert len(obs["messages"]) == 1
    assert "persona_turns" not in info


def test_the_cast_is_announced_on_reset():
    env = builder(population(persona("nurse", wake_on=[])), ReplyDriver()).build()
    _obs, info = env.reset(seed=1)
    assert info["personas"] == [
        {"id": "nurse", "name": "Nurse", "role": "charge nurse"}
    ]


def test_a_colleague_replies_within_the_same_step_the_agent_asked():
    env = builder(
        population(persona("nurse", wake_on=["post_message"], latency_steps=0)),
        ReplyDriver(),
    ).build()
    env.reset(seed=1)
    obs, _, _, _, info = env.step({"type": "post_message", "body": "nurse?"})
    authors = {m["author"] for m in obs["messages"].values()}
    assert authors == {"agent", "nurse"}
    assert info["persona_turns"][0]["persona_id"] == "nurse"


def test_persona_writes_land_in_the_step_diff_not_the_next_one():
    env = builder(
        population(persona("nurse", wake_on=["post_message"])), ReplyDriver()
    ).build()
    env.reset(seed=1)
    env.step({"type": "post_message", "body": "nurse?"})
    step = env.current_trajectory().steps[0]
    assert len(step.diff["added"]) or len(step.diff["changed"])
    assert len(step.events) == 3  # agent's, the persona's, and the turn marker


def test_a_persona_who_is_not_woken_stays_quiet():
    env = builder(
        population(persona("nurse", wake_on=["page_nurse"], activity=0)), ReplyDriver()
    ).build()
    env.reset(seed=1)
    obs, _, _, _, info = env.step({"type": "review_chart"})
    assert obs["messages"] == {}
    assert "persona_turns" not in info


def test_latency_makes_the_reply_arrive_a_step_later():
    env = builder(
        population(
            persona(
                "nurse",
                wake_on=["post_message"],
                latency_steps=2,
                responsiveness=0,
                activity=0,
            )
        ),
        ReplyDriver(),
    ).build()
    env.reset(seed=1)
    obs, _, _, _, _ = env.step({"type": "post_message", "body": "nurse?"})
    assert len(obs["messages"]) == 1
    obs, _, _, _, _ = env.step({"type": "review_chart"})
    assert len(obs["messages"]) == 1
    obs, _, _, _, _ = env.step({"type": "review_chart"})
    assert len(obs["messages"]) == 2


def test_a_persona_who_leaves_their_action_space_changes_nothing():
    class RogueDriver(PersonaDriver):
        def act(self, view: PersonaView) -> PersonaTurn:
            return PersonaTurn(
                persona_id=view.persona.id, action=Action(type="review_chart")
            )

    env = builder(
        population(persona("nurse", wake_on=[], allowed_actions=["post_message"])),
        RogueDriver(),
    ).build()
    env.reset(seed=1)
    obs, _, _, _, info = env.step({"type": "post_message", "body": "hi"})
    assert obs["chart"]["c_0"]["reviewed"] == 0
    assert info["persona_turns"][0]["blocked"]


def test_building_with_an_action_the_environment_lacks_fails_at_build_time():
    pop = population(persona("nurse", allowed_actions=["send_page"]))
    with pytest.raises(EnvironmentBuildError, match="send_page"):
        builder(pop, ReplyDriver()).build()


def test_the_error_names_the_actions_that_do_exist():
    pop = population(persona("nurse", allowed_actions=["send_page"]))
    with pytest.raises(EnvironmentBuildError, match="post_message"):
        builder(pop, ReplyDriver()).build()


def test_archetype_action_spaces_are_validated_too():
    pop = population(
        persona("nurse"), archetypes=[persona("patient", allowed_actions=["nope"])]
    )
    with pytest.raises(EnvironmentBuildError, match="nope"):
        builder(pop, ReplyDriver()).build()


def test_the_determinism_check_passes_with_a_cast_present():
    """Built with verify=True by default — this failing would raise."""
    env = builder(
        population(persona("nurse", wake_on=[], activity=50)), ReplyDriver()
    ).build()
    assert env.personas.enabled


def test_a_model_backed_cast_still_passes_the_build_time_determinism_probe():
    """A model driver cannot be replayed; the probe swaps in the scripted one."""
    calls = []

    class NondeterministicDriver(PersonaDriver):
        def act(self, view: PersonaView) -> PersonaTurn:
            calls.append(view.persona.id)
            return PersonaTurn(
                persona_id=view.persona.id,
                action=Action(
                    type="post_message", params={"body": f"reply {len(calls)}"}
                ),
            )

    env = builder(
        population(persona("nurse", wake_on=[])), NondeterministicDriver()
    ).build()
    assert calls == [], "the probe must not have used the nondeterministic driver"
    # ...and the real driver is restored for actual episodes.
    env.reset(seed=1)
    env.step({"type": "post_message", "body": "hi"})
    assert calls == ["nurse"]


def test_two_same_seed_episodes_produce_the_same_persona_behavior():
    def run():
        env = builder(
            population(
                persona("nurse", wake_on=["never"], activity=60, cooldown_steps=0)
            ),
            ReplyDriver(),
        ).build(verify=False)
        env.reset(seed=5)
        authors = []
        for _ in range(6):
            obs, _, _, _, _ = env.step({"type": "review_chart"})
            authors.append(sorted(m["author"] for m in obs["messages"].values()))
        return authors

    assert run() == run()


def test_a_persona_write_is_visible_to_the_final_grade():
    env = builder(
        population(persona("nurse", wake_on=["post_message"])), ReplyDriver()
    ).build()
    env.reset(seed=1)
    env.step({"type": "post_message", "body": "nurse?"})
    evaluation = env.finalize_episode("test")
    assert evaluation is not None
    assert len(env.state.get()["messages"]) == 2

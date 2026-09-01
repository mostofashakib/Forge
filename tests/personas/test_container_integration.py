"""Personas inside a containerized environment.

The container family reaches its world over HTTP, so the thing worth proving
here is that a persona's action takes the same wire path the agent's does —
one POST through the env's transport, not a side channel.
"""
from __future__ import annotations

import httpx

from forge.contracts.persona import PersonaDriver, PersonaTurn, PersonaView
from forge.contracts.types import Action
from forge.envgen.container_env_base import ContainerEnvBase
from forge.personas.engine import PersonaEngine

from tests.personas.conftest import persona, population


class ReplyDriver(PersonaDriver):
    def act(self, view: PersonaView) -> PersonaTurn:
        return PersonaTurn(
            persona_id=view.persona.id,
            trigger=view.trigger,
            action=Action(type="post_message", params={"body": "on it"}),
        )


def make_env(pop=None, driver=None):
    """A container whose /forge/state counts every action it was posted."""
    posted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/forge/state":
            return httpx.Response(200, json={"posts": len(posted)})
        if path.startswith("/forge/"):
            return httpx.Response(200, json={})
        posted.append(path)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    engine = (
        PersonaEngine(pop, driver=driver, environment_actions=["post_message"])
        if pop is not None
        else None
    )
    env = ContainerEnvBase("http://env", client=client, personas=engine)
    return env, posted


def test_a_container_without_personas_posts_only_the_agent_action():
    env, posted = make_env()
    env.reset(seed=1)
    env.step({"type": "post_message"})
    assert posted == ["/post_message"]


def test_a_persona_action_goes_over_the_same_transport():
    env, posted = make_env(population(persona("nurse", wake_on=[])), ReplyDriver())
    env.reset(seed=1)
    env.step({"type": "post_message"})
    assert posted == ["/post_message", "/post_message"]


def test_the_cast_is_announced_on_reset():
    env, _ = make_env(population(persona("nurse", wake_on=[])), ReplyDriver())
    _obs, info = env.reset(seed=1)
    assert info["personas"] == [
        {"id": "nurse", "name": "Nurse", "role": "charge nurse"}
    ]


def test_the_observation_reflects_the_state_after_the_cast_acted():
    env, _ = make_env(population(persona("nurse", wake_on=[])), ReplyDriver())
    env.reset(seed=1)
    obs, _, _, _, info = env.step({"type": "post_message"})
    assert obs["posts"] == 2
    assert info["persona_turns"][0]["persona_id"] == "nurse"


def test_a_blocked_persona_action_never_reaches_the_container():
    class RogueDriver(PersonaDriver):
        def act(self, view: PersonaView) -> PersonaTurn:
            return PersonaTurn(
                persona_id=view.persona.id, action=Action(type="drop_database")
            )

    env, posted = make_env(
        population(persona("nurse", wake_on=[], allowed_actions=["post_message"])),
        RogueDriver(),
    )
    env.reset(seed=1)
    _obs, _, _, _, info = env.step({"type": "post_message"})
    assert posted == ["/post_message"]
    assert info["persona_turns"][0]["blocked"]

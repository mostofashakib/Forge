"""Shared fixtures for the persona suite."""
from __future__ import annotations

import copy

import pytest

from forge.contracts import InitialStateProvider, ToolParam, ToolSpec
from forge.contracts.persona import (
    PersonaBehavior,
    PersonaPopulation,
    PersonaProfile,
    PersonaSpec,
    PersonaTraits,
)
from forge.runtime.transition import TransitionResult


class MessageBoardFactory(InitialStateProvider):
    """A ward noticeboard: anyone in the room can post to it."""

    def reset(self, ctx, *, seed: int | None, options: dict) -> dict:
        return {"messages": {}, "chart": {"c_0": {"id": "c_0", "reviewed": 0}}}


def post_message(state, action, ctx):
    new_state = copy.deepcopy(state)
    message_id = ctx.id_generator.next("msg")
    new_state["messages"][message_id] = {
        "id": message_id,
        "author": ctx.actor_id,
        "body": action.get("body", ""),
    }
    return TransitionResult(
        state=new_state,
        events=[{"type": "message_posted", "entity_id": message_id}],
    )


def review_chart(state, action, ctx):
    new_state = copy.deepcopy(state)
    new_state["chart"]["c_0"]["reviewed"] += 1
    return TransitionResult(
        state=new_state, events=[{"type": "chart_reviewed", "entity_id": "chart"}]
    )


POST_MESSAGE_SPEC = ToolSpec(
    name="post_message",
    description="Write a note on the ward board",
    params=[ToolParam(name="body", type="string", description="what to say")],
)


def persona(
    persona_id: str = "nurse",
    *,
    allowed_actions: list[str] | None = None,
    wake_on: list[str] | None = None,
    activity: int = 0,
    latency_steps: int = 0,
    cooldown_steps: int = 1,
    max_actions_per_episode: int | None = None,
    responsiveness: int = 50,
    initiative: int = 30,
) -> PersonaSpec:
    return PersonaSpec(
        profile=PersonaProfile(
            id=persona_id,
            name=persona_id.title(),
            role="charge nurse",
            traits=PersonaTraits(responsiveness=responsiveness, initiative=initiative),
        ),
        behavior=PersonaBehavior(
            allowed_actions=allowed_actions if allowed_actions is not None else ["post_message"],
            wake_on=wake_on or [],
            activity=activity,
            latency_steps=latency_steps,
            cooldown_steps=cooldown_steps,
            max_actions_per_episode=max_actions_per_episode,
        ),
    )


def population(*specs: PersonaSpec, **kwargs) -> PersonaPopulation:
    kwargs.setdefault("enabled", True)
    return PersonaPopulation(roster=list(specs), **kwargs)


@pytest.fixture
def nurse() -> PersonaSpec:
    return persona()

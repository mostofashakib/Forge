"""Keeping a model-backed persona in character."""
from __future__ import annotations

from forge.contracts.persona import PersonaProfile, PersonaSpec, PersonaTraits
from forge.contracts.types import Observation, Task
from forge.personas.archetypes import ARCHETYPES, archetype, archetype_ids
from forge.personas.prompting import PersonaPromptTemplate, describe_traits

from tests.personas.conftest import POST_MESSAGE_SPEC, persona


def template(spec=None):
    return PersonaPromptTemplate(
        spec or persona("nurse", allowed_actions=["post_message"])
    )


def task():
    return Task(id="t", objective="cover the shift")


def test_the_system_prompt_names_the_person_and_their_role():
    text = template().system(task())
    assert "Nurse" in text
    assert "charge nurse" in text


def test_the_system_prompt_states_the_action_space_explicitly():
    text = template().system(task())
    assert "post_message" in text
    assert "only things you are able to do" in text


def test_the_system_prompt_forbids_inventing_actions():
    assert "invent an action" in template().system(task())


def test_staying_quiet_is_offered_as_a_legitimate_choice():
    """A persona who must act every turn is not a person."""
    assert "staying quiet" in template().system(task())


def test_private_knowledge_reaches_the_persona_and_nothing_else():
    spec = PersonaSpec(
        profile=PersonaProfile(
            id="p", name="Priya", knowledge={"allergy": "penicillin"}
        )
    )
    assert "penicillin" in PersonaPromptTemplate(spec).system(task())


def test_traits_are_rendered_as_instructions_not_numbers():
    spec = PersonaSpec(
        profile=PersonaProfile(
            id="p", name="P", traits=PersonaTraits(verbosity=5, formality=5)
        )
    )
    text = PersonaPromptTemplate(spec).system(task())
    assert "verbosity" not in text.lower()
    assert "sentence or two" in text


def test_opposite_trait_values_produce_opposite_instructions():
    terse = PersonaSpec(profile=PersonaProfile(id="a", name="A", traits=PersonaTraits(verbosity=0)))
    wordy = PersonaSpec(profile=PersonaProfile(id="b", name="B", traits=PersonaTraits(verbosity=100)))
    assert describe_traits(terse) != describe_traits(wordy)


def test_the_observation_is_rendered_stably():
    """Same situation, same prompt — an env is only reproducible if its text is."""
    tmpl = template()
    obs = Observation(payload={"b": 2, "a": 1})
    assert tmpl.user(obs, task()) == tmpl.user(Observation(payload={"a": 1, "b": 2}), task())


def test_tool_descriptions_carry_the_action_parameters():
    described = template().tool_descriptions([POST_MESSAGE_SPEC])
    assert described[0]["name"] == "post_message"
    assert "body" in described[0]["input_schema"]["properties"]


# --- archetypes -----------------------------------------------------------


def test_every_archetype_ships_inert():
    """An archetype used as-is must not be able to act until it is bound."""
    for name, spec in ARCHETYPES.items():
        assert spec.behavior.allowed_actions == [], name


def test_binding_an_archetype_gives_it_an_action_space():
    spec = archetype("meticulous_checker", allowed_actions=["post_message"])
    assert spec.behavior.allowed_actions == ["post_message"]
    assert spec.profile.traits.diligence == 95


def test_binding_does_not_mutate_the_library_entry():
    archetype("meticulous_checker", allowed_actions=["post_message"])
    assert ARCHETYPES["meticulous_checker"].behavior.allowed_actions == []


def test_an_unknown_archetype_lists_the_available_ones():
    import pytest

    with pytest.raises(KeyError, match="meticulous_checker"):
        archetype("space_pirate")


def test_archetype_ids_are_sorted():
    assert archetype_ids() == sorted(ARCHETYPES)

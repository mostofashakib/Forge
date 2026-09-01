"""Resolving a configured population into an episode's concrete cast."""
from __future__ import annotations

import pytest

from forge.contracts.persona import PersonaPopulation
from forge.personas.population import (
    PersonaPopulationError,
    population_seed,
    resolve_roster,
)

from tests.personas.conftest import persona


def test_disabled_population_produces_no_cast():
    pop = PersonaPopulation(enabled=False, roster=[persona("nurse")], count=3)
    assert resolve_roster(pop, seed=1) == []


def test_explicit_roster_keeps_configured_order():
    pop = PersonaPopulation(
        enabled=True, roster=[persona("nurse"), persona("doctor"), persona("porter")]
    )
    assert [s.profile.id for s in resolve_roster(pop, seed=1)] == [
        "nurse",
        "doctor",
        "porter",
    ]


def test_count_below_roster_truncates_rather_than_shuffling():
    pop = PersonaPopulation(
        enabled=True,
        count=2,
        roster=[persona("nurse"), persona("doctor"), persona("porter")],
    )
    assert [s.profile.id for s in resolve_roster(pop, seed=7)] == ["nurse", "doctor"]


def test_count_above_roster_is_filled_from_archetypes():
    pop = PersonaPopulation(
        enabled=True,
        count=4,
        roster=[persona("nurse")],
        archetypes=[persona("patient")],
    )
    roster = resolve_roster(pop, seed=7)
    assert len(roster) == 4
    assert roster[0].profile.id == "nurse"
    assert all(s.profile.id.startswith("patient_") for s in roster[1:])


def test_clones_never_share_an_id_or_a_name():
    pop = PersonaPopulation(
        enabled=True, count=6, roster=[], archetypes=[persona("patient")]
    )
    roster = resolve_roster(pop, seed=3)
    ids = [s.profile.id for s in roster]
    names = [s.profile.name for s in roster]
    assert len(set(ids)) == len(ids), ids
    assert len(set(names)) == len(names), names


def test_clone_id_does_not_collide_with_an_explicit_roster_id():
    """A roster entry already named `patient_1` must not be duplicated."""
    pop = PersonaPopulation(
        enabled=True,
        count=3,
        roster=[persona("patient_1")],
        archetypes=[persona("patient")],
    )
    ids = [s.profile.id for s in resolve_roster(pop, seed=3)]
    assert len(set(ids)) == len(ids), ids


def test_clones_inherit_the_archetype_action_space():
    pop = PersonaPopulation(
        enabled=True,
        count=2,
        archetypes=[persona("patient", allowed_actions=["post_message"])],
    )
    roster = resolve_roster(pop, seed=3)
    assert all(s.behavior.allowed_actions == ["post_message"] for s in roster)


def test_same_seed_produces_an_identical_cast():
    pop = PersonaPopulation(
        enabled=True, count=5, archetypes=[persona("patient"), persona("visitor")]
    )
    first = [(s.profile.id, s.profile.name) for s in resolve_roster(pop, seed=99)]
    second = [(s.profile.id, s.profile.name) for s in resolve_roster(pop, seed=99)]
    assert first == second


def test_different_seeds_produce_different_names():
    """Negative control for the test above: seeding must actually do something."""
    pop = PersonaPopulation(
        enabled=True, count=5, archetypes=[persona("patient")]
    )
    first = [s.profile.name for s in resolve_roster(pop, seed=1)]
    second = [s.profile.name for s in resolve_roster(pop, seed=2)]
    assert first != second


def test_count_beyond_roster_without_archetypes_is_an_error():
    pop = PersonaPopulation(enabled=True, count=4, roster=[persona("nurse")])
    with pytest.raises(PersonaPopulationError, match="no archetypes"):
        resolve_roster(pop, seed=1)


def test_resolving_does_not_mutate_the_configured_roster():
    original = persona("nurse")
    pop = PersonaPopulation(enabled=True, roster=[original])
    resolved = resolve_roster(pop, seed=1)
    resolved[0].behavior.allowed_actions.append("mutated")
    assert original.behavior.allowed_actions == ["post_message"]


def test_pinned_seed_overrides_the_episode_seed():
    pop = PersonaPopulation(enabled=True, seed=1234)
    assert population_seed(pop, 1) == 1234
    assert population_seed(pop, 999) == 1234


def test_unpinned_seed_is_derived_from_but_not_equal_to_the_episode_seed():
    pop = PersonaPopulation(enabled=True)
    assert population_seed(pop, 42) != 42
    assert population_seed(pop, 42) == population_seed(pop, 42)
    assert population_seed(pop, 42) != population_seed(pop, 43)

"""Turning a configured population into the concrete cast for one episode.

Resolution is pure and deterministic: the same population and seed always
produce the same roster, in the same order, with the same ids. Nothing here
touches the environment, so a caller can preview the cast a seed will produce
without running an episode — which is what the configuration UI does.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from forge.contracts.persona import PersonaPopulation, PersonaProfile, PersonaSpec

# Filler names for archetype clones. A generated persona is still a person to
# the agent, so it gets a name rather than "nurse_2". The list is fixed and
# indexed deterministically, never sampled with replacement, so two clones of
# the same archetype never collide on a name until the list is exhausted.
_FILLER_NAMES = (
    "Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper", "Indigo",
    "Jordan", "Kai", "Lennox", "Marlowe", "Noor", "Oakley", "Payton", "Quinn",
    "Reese", "Sawyer", "Tatum", "Umi", "Vale", "Wren", "Xen", "Yael", "Zephyr",
)


class PersonaPopulationError(ValueError):
    """A population that cannot produce the cast it was asked for."""


def population_seed(population: PersonaPopulation, episode_seed: int) -> int:
    """The seed persona machinery draws from.

    Pinned by `population.seed` when set, so a cast and its cadence can be held
    fixed while the environment's own seed varies. Otherwise derived from the
    episode seed by an offset, so persona randomness is reproducible without
    being *identical* to the environment's stream — two independent streams
    that happen to share a seed produce correlated draws, and correlated
    persona timing is a real source of confounded rollouts.
    """
    if population.seed is not None:
        return population.seed
    return (episode_seed * 2_654_435_761 + 0x9E3779B9) % (2**31)


def resolve_roster(
    population: PersonaPopulation, seed: int
) -> list[PersonaSpec]:
    """The concrete cast for an episode.

    The explicit roster comes first and keeps its configured order. When
    `count` asks for more than the roster holds, archetypes are cloned to fill
    the gap; when it asks for fewer, the roster is truncated from the end.
    """
    if not population.enabled:
        return []

    roster = [spec.model_copy(deep=True) for spec in population.roster]
    target = population.count if population.count is not None else len(roster)

    if target <= len(roster):
        return roster[:target]

    shortfall = target - len(roster)
    if not population.archetypes:
        raise PersonaPopulationError(
            f"population asks for {target} personas but supplies only "
            f"{len(roster)} in its roster and no archetypes to fill the "
            f"remaining {shortfall}"
        )

    taken = {spec.profile.id for spec in roster}
    used_names = {spec.profile.name for spec in roster}
    rng = random.Random(seed)
    for index in range(shortfall):
        archetype = population.archetypes[index % len(population.archetypes)]
        roster.append(_clone(archetype, index, taken, used_names, rng))
    return roster


def _clone(
    archetype: PersonaSpec,
    index: int,
    taken: set[str],
    used_names: set[str],
    rng: random.Random,
) -> PersonaSpec:
    """One archetype instance, with a unique id and a name of its own."""
    clone = archetype.model_copy(deep=True)
    base_id = archetype.profile.id or "persona"
    candidate = f"{base_id}_{index + 1}"
    suffix = index + 1
    while candidate in taken:
        suffix += 1
        candidate = f"{base_id}_{suffix}"
    taken.add(candidate)

    profile = clone.profile.model_copy(
        update={"id": candidate, "name": _name_for(archetype.profile, used_names, rng)}
    )
    return PersonaSpec(profile=profile, behavior=clone.behavior)


def _name_for(
    archetype: PersonaProfile, used_names: set[str], rng: random.Random
) -> str:
    """A human name not already in use, drawn deterministically."""
    available = [name for name in _FILLER_NAMES if name not in used_names]
    if not available:
        # More clones than filler names. Fall back to numbering rather than
        # repeating a name — two colleagues with the same name is a confusion
        # the agent should never have to resolve.
        chosen = f"{archetype.name or 'Colleague'} {len(used_names) + 1}"
    else:
        chosen = available[rng.randrange(len(available))]
    used_names.add(chosen)
    return chosen


def roster_ids(roster: Sequence[PersonaSpec]) -> list[str]:
    return [spec.profile.id for spec in roster]

"""Reading and writing the `personas:` block of an environment's config.

The wire format is flat where a human writes it and nested where the shape
genuinely differs: identity fields sit at the top of a roster entry, and the
engagement dials sit under `behavior:`, because "who is this" and "how do they
join in" are separately edited and separately reviewed.

An entry may name an `archetype:`, in which case the library entry supplies
every field the entry does not. That is the difference between describing a
persona and adjusting one.

Unknown keys are rejected rather than ignored. A misspelled `respnsiveness`
that silently defaults to 50 produces an environment that behaves nothing like
the one its author configured, and no error to explain why.
"""
from __future__ import annotations

from typing import Any

from forge.contracts.persona import (
    PersonaBehavior,
    PersonaPopulation,
    PersonaProfile,
    PersonaSpec,
    PersonaTraits,
)
from forge.personas.archetypes import ARCHETYPES, archetype_ids

_PROFILE_KEYS = {"id", "name", "role", "backstory", "goals", "traits", "knowledge", "style"}
_ENTRY_KEYS = _PROFILE_KEYS | {"archetype", "behavior"}
_POPULATION_KEYS = {
    "enabled",
    "count",
    "roster",
    "archetypes",
    "max_actions_per_step",
    "seed",
    "driver",
}


class PersonaConfigError(ValueError):
    """A `personas:` block that cannot be read."""


def _reject_unknown(raw: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise PersonaConfigError(
            f"unknown {where} field(s): {', '.join(unknown)}. "
            f"Valid fields: {', '.join(sorted(allowed))}"
        )


def _traits(raw: Any, base: PersonaTraits) -> PersonaTraits:
    if raw is None:
        return base
    if not isinstance(raw, dict):
        raise PersonaConfigError(f"traits must be a mapping, got {type(raw).__name__}")
    _reject_unknown(raw, set(PersonaTraits.model_fields), "trait")
    try:
        # Validated rather than copied, so an out-of-range dial is caught here
        # with the field named — `model_copy` skips validation entirely.
        return PersonaTraits.model_validate({**base.model_dump(), **raw})
    except (TypeError, ValueError) as exc:
        raise PersonaConfigError(f"invalid traits: {exc}") from exc


def _behavior(raw: Any, base: PersonaBehavior) -> PersonaBehavior:
    if raw is None:
        return base
    if not isinstance(raw, dict):
        raise PersonaConfigError(f"behavior must be a mapping, got {type(raw).__name__}")
    _reject_unknown(raw, set(PersonaBehavior.model_fields), "behavior")
    try:
        return PersonaBehavior.model_validate({**base.model_dump(), **raw})
    except ValueError as exc:
        raise PersonaConfigError(f"invalid behavior: {exc}") from exc


def load_spec(raw: dict, index: int = 0) -> PersonaSpec:
    """One roster entry, with an archetype supplying whatever it omits."""
    if not isinstance(raw, dict):
        raise PersonaConfigError(
            f"persona entry {index} must be a mapping, got {type(raw).__name__}"
        )
    _reject_unknown(raw, _ENTRY_KEYS, "persona")

    base_id = raw.get("archetype")
    if base_id is not None:
        if base_id not in ARCHETYPES:
            raise PersonaConfigError(
                f"unknown archetype {base_id!r}; available: {', '.join(archetype_ids())}"
            )
        base = ARCHETYPES[base_id].model_copy(deep=True)
    else:
        base = PersonaSpec(
            profile=PersonaProfile(id=f"persona_{index + 1}", name=f"Persona {index + 1}")
        )

    profile_update = {
        key: raw[key] for key in _PROFILE_KEYS if key in raw and key != "traits"
    }
    profile = base.profile.model_copy(update=profile_update)
    profile = profile.model_copy(update={"traits": _traits(raw.get("traits"), base.profile.traits)})
    if not profile.id:
        raise PersonaConfigError(f"persona entry {index} has no id")

    try:
        profile = PersonaProfile.model_validate(profile.model_dump())
    except ValueError as exc:
        raise PersonaConfigError(f"invalid persona {profile.id!r}: {exc}") from exc

    return PersonaSpec(profile=profile, behavior=_behavior(raw.get("behavior"), base.behavior))


def load_population(raw: Any) -> PersonaPopulation:
    """The `personas:` block as a `PersonaPopulation`.

    A missing or empty block yields a disabled population rather than an error,
    so every existing environment keeps loading unchanged.
    """
    if not raw:
        return PersonaPopulation()
    if not isinstance(raw, dict):
        raise PersonaConfigError(
            f"personas must be a mapping, got {type(raw).__name__}"
        )
    _reject_unknown(raw, _POPULATION_KEYS, "personas")

    roster = [load_spec(entry, i) for i, entry in enumerate(raw.get("roster") or [])]
    archetypes = [
        load_spec(entry, i) for i, entry in enumerate(raw.get("archetypes") or [])
    ]
    duplicates = _duplicate_ids(roster)
    if duplicates:
        raise PersonaConfigError(
            f"duplicate persona id(s) in roster: {', '.join(duplicates)}"
        )

    try:
        return PersonaPopulation(
            enabled=bool(raw.get("enabled", False)),
            count=raw.get("count"),
            roster=roster,
            archetypes=archetypes,
            max_actions_per_step=int(raw.get("max_actions_per_step", 1)),
            seed=raw.get("seed"),
            driver=str(raw.get("driver", "scripted")),
        )
    except ValueError as exc:
        raise PersonaConfigError(f"invalid personas block: {exc}") from exc


def _duplicate_ids(roster: list[PersonaSpec]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for spec in roster:
        if spec.profile.id in seen:
            duplicates.add(spec.profile.id)
        seen.add(spec.profile.id)
    return sorted(duplicates)


def dump_spec(spec: PersonaSpec) -> dict:
    """One roster entry, in the same shape `load_spec` reads.

    Fully expanded — no `archetype:` reference — because a round trip through
    the editor should never depend on a library entry that may change under it.
    """
    profile = spec.profile
    entry: dict = {"id": profile.id, "name": profile.name}
    if profile.role:
        entry["role"] = profile.role
    if profile.backstory:
        entry["backstory"] = profile.backstory
    if profile.goals:
        entry["goals"] = list(profile.goals)
    if profile.style:
        entry["style"] = profile.style
    if profile.knowledge:
        entry["knowledge"] = dict(profile.knowledge)
    entry["traits"] = profile.traits.model_dump()
    entry["behavior"] = spec.behavior.model_dump()
    return entry


def dump_population(population: PersonaPopulation) -> dict:
    """A `PersonaPopulation` as the mapping that belongs under `personas:`."""
    payload: dict = {
        "enabled": population.enabled,
        "driver": population.driver,
        "max_actions_per_step": population.max_actions_per_step,
    }
    if population.count is not None:
        payload["count"] = population.count
    if population.seed is not None:
        payload["seed"] = population.seed
    payload["roster"] = [dump_spec(spec) for spec in population.roster]
    if population.archetypes:
        payload["archetypes"] = [dump_spec(spec) for spec in population.archetypes]
    return payload

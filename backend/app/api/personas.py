# backend/app/api/personas.py
"""Configuring the simulated humans that share an environment with the agent.

The cast lives in the `personas:` block of an environment's
`custom/config.yaml` — the same file the YAML editor edits — rather than in a
file of its own. One source of truth means the two editors can never disagree,
and an author who prefers YAML keeps working exactly as before.

Reads and writes go through `forge.personas.config`, so the API rejects a
misspelled trait or an out-of-range dial with the same message the runtime
would, instead of accepting it and producing an environment that behaves
nothing like the one that was configured.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from forge.contracts.persona import PersonaPopulation, PersonaSpec
from forge.personas.archetypes import ARCHETYPES, archetype_ids
from forge.personas.config import (
    PersonaConfigError,
    dump_population,
    dump_spec,
    load_population,
)
from forge.personas.population import PersonaPopulationError, resolve_roster
from forge.settings import generated_envs_root

router = APIRouter(prefix="/api/envs")


def _validate_env_name(env_name: str) -> None:
    if ".." in env_name or "/" in env_name or "\\" in env_name:
        raise HTTPException(status_code=400, detail="Invalid environment name")


def _config_path(env_name: str) -> Path:
    _validate_env_name(env_name)
    custom_dir = generated_envs_root() / env_name / "custom"
    if not custom_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"Environment '{env_name}' not found"
        )
    return custom_dir / "config.yaml"


def _read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(
            status_code=422, detail=f"config.yaml is not valid YAML: {exc}"
        )


class PersonaPayload(BaseModel):
    """The cast, plus what the editor needs to render controls for it."""

    personas: dict
    environment_actions: list[str] = Field(default_factory=list)
    archetypes: list[dict] = Field(default_factory=list)


class PersonaUpdate(BaseModel):
    personas: dict


class PreviewRequest(BaseModel):
    """Which cast a given seed would actually produce."""

    personas: dict
    seed: int = 0


class PreviewResponse(BaseModel):
    roster: list[dict]
    warnings: list[str] = Field(default_factory=list)


def _environment_actions(env_name: str) -> list[str]:
    """The action types this environment implements.

    Read from the generated `transitions/` directory, which holds one module
    per action named after it, rather than by importing the environment —
    importing would mean executing generated code inside the API process. An
    environment whose surface cannot be read yields an empty list, and the
    editor degrades to free text rather than failing.
    """
    transitions = generated_envs_root() / env_name / "transitions"
    if not transitions.is_dir():
        return []
    return sorted(
        path.stem
        for path in transitions.glob("*.py")
        if not path.stem.startswith("_")
    )


def _archetype_catalog() -> list[dict]:
    return [
        {
            "id": archetype_id,
            **{
                key: value
                for key, value in dump_spec(ARCHETYPES[archetype_id]).items()
                if key != "behavior"
            },
            "behavior": ARCHETYPES[archetype_id].behavior.model_dump(),
        }
        for archetype_id in archetype_ids()
    ]


@router.get("/{env_name}/personas", response_model=PersonaPayload)
def get_personas(env_name: str) -> PersonaPayload:
    path = _config_path(env_name)
    raw = _read_config(path)
    try:
        population = load_population(raw.get("personas"))
    except PersonaConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return PersonaPayload(
        personas=dump_population(population),
        environment_actions=_environment_actions(env_name),
        archetypes=_archetype_catalog(),
    )


@router.put("/{env_name}/personas", response_model=PersonaPayload)
def update_personas(env_name: str, payload: PersonaUpdate) -> PersonaPayload:
    path = _config_path(env_name)
    try:
        population = load_population(payload.personas)
    except PersonaConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    _reject_unknown_actions(env_name, population)

    raw = _read_config(path)
    # Rewritten in place: everything else in config.yaml — reward weights,
    # observation filtering — must survive an edit made from the persona page.
    raw["personas"] = dump_population(population)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return PersonaPayload(
        personas=raw["personas"],
        environment_actions=_environment_actions(env_name),
        archetypes=_archetype_catalog(),
    )


@router.post("/{env_name}/personas/preview", response_model=PreviewResponse)
def preview_personas(env_name: str, payload: PreviewRequest) -> PreviewResponse:
    """The cast a seed produces, without running an episode.

    `count` above the roster size clones archetypes, so what an author
    configures and what an episode actually contains are not the same list.
    Showing the resolved cast is the difference between configuring four
    personas and discovering four personas.
    """
    _validate_env_name(env_name)
    try:
        population = load_population(payload.personas)
    except PersonaConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        roster = resolve_roster(population, payload.seed)
    except PersonaPopulationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return PreviewResponse(
        roster=[dump_spec(spec) for spec in roster],
        warnings=_warnings(env_name, roster),
    )


def _warnings(env_name: str, roster: list[PersonaSpec]) -> list[str]:
    """Configuration that is valid but will not do what its author expects.

    Distinct from the errors that block a save: these are castings that load,
    run, and quietly do nothing — the failure mode an author is least likely to
    diagnose on their own.
    """
    warnings: list[str] = []
    known = set(_environment_actions(env_name))
    for spec in roster:
        name = spec.profile.name
        if not spec.behavior.allowed_actions:
            warnings.append(f"{name} has no allowed actions and will never act.")
            continue
        if known:
            unknown = sorted(set(spec.behavior.allowed_actions) - known)
            if unknown:
                warnings.append(
                    f"{name} is allowed {', '.join(unknown)}, which this "
                    "environment does not implement."
                )
        if spec.behavior.activity == 0 and not spec.behavior.wake_on:
            # Reachable only by an agent action, and `wake_on: []` means any
            # action wakes them — so this one is fine. Nothing to warn about.
            continue
        if spec.behavior.activity == 0 and spec.behavior.max_actions_per_episode == 0:
            warnings.append(
                f"{name} has a zero action budget and will never act."
            )
    return warnings


def _reject_unknown_actions(env_name: str, population: PersonaPopulation) -> None:
    """Refuse to save a cast bound to actions the environment does not have.

    The same check `EnvBuilder.build()` makes. Catching it here means an author
    sees the mistake while editing rather than when the next rollout starts.
    """
    known = set(_environment_actions(env_name))
    if not known:
        # The surface could not be read. Saving is still allowed — the build
        # will catch it — rather than blocking on our inability to introspect.
        return
    for spec in [*population.roster, *population.archetypes]:
        unknown = sorted(set(spec.behavior.allowed_actions) - known)
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"persona '{spec.profile.id}' is allowed action(s) "
                    f"{', '.join(unknown)}, which '{env_name}' does not "
                    f"implement. Available: {', '.join(sorted(known))}"
                ),
            )

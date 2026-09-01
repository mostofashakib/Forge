"""Simulated humans that share an environment with the agent.

An RL environment that models only records and forms teaches an agent to edit
records and fill forms. Most of the work worth learning is not that — it is a
handover to a colleague who is busy, a question answered three steps later, a
nurse who tells you the chart is wrong. This package puts those people in the
environment.

The contract lives in `forge/contracts/persona.py`; everything here is an
implementation of it:

    from forge.personas import PersonaEngine, load_population

    population = load_population(yaml.safe_load(config_text))
    engine = PersonaEngine(population, environment_actions=env.action_types)

`EnvBuilder.with_personas(...)` wires one into an in-process environment, and
`ContainerEnvBase(personas=...)` into a containerized one. CLI and browser
environments do not take a cast: there is no coherent notion of a colleague
inside a shell session.
"""
from forge.contracts.persona import (
    PersonaBehavior,
    PersonaDriver,
    PersonaPopulation,
    PersonaProfile,
    PersonaScheduler,
    PersonaSpec,
    PersonaTick,
    PersonaTraits,
    PersonaTurn,
    PersonaView,
)
from forge.personas.archetypes import ARCHETYPES, archetype, archetype_ids
from forge.personas.config import dump_population, load_population
from forge.personas.drivers import (
    AgentPersonaDriver,
    ScriptedPersonaDriver,
    make_driver,
)
from forge.personas.engine import PersonaEngine, PersonaTickResult
from forge.personas.guardrails import ActionGuard, GuardDecision
from forge.personas.population import (
    PersonaPopulationError,
    population_seed,
    resolve_roster,
)
from forge.personas.prompting import PersonaPromptTemplate, describe_traits
from forge.personas.scheduler import PersonaScheduleState, TraitScheduler

__all__ = [
    "ARCHETYPES",
    "ActionGuard",
    "AgentPersonaDriver",
    "GuardDecision",
    "PersonaBehavior",
    "PersonaDriver",
    "PersonaEngine",
    "PersonaPopulation",
    "PersonaPopulationError",
    "PersonaProfile",
    "PersonaPromptTemplate",
    "PersonaScheduleState",
    "PersonaScheduler",
    "PersonaSpec",
    "PersonaTick",
    "PersonaTickResult",
    "PersonaTraits",
    "PersonaTurn",
    "PersonaView",
    "ScriptedPersonaDriver",
    "TraitScheduler",
    "archetype",
    "archetype_ids",
    "describe_traits",
    "dump_population",
    "load_population",
    "make_driver",
    "population_seed",
    "resolve_roster",
]

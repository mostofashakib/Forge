"""What a persona does once it has been given a turn.

Two implementations, and the difference between them is the whole point of the
design. `ScriptedPersonaDriver` is deterministic and free; `AgentPersonaDriver`
hands the decision to a model so the persona behaves like a person rather than
a rotation through a list. Both are bounded by the same guardrail, so switching
between them changes how an environment *feels* without changing what its
inhabitants are permitted to do.

Neither driver is trusted. A driver returns a proposal; `PersonaEngine` decides
whether it runs.
"""
from __future__ import annotations

import random
from collections.abc import Callable

from forge.contracts.persona import PersonaDriver, PersonaTurn, PersonaView
from forge.contracts.types import Action, Task
from forge.personas.prompting import PersonaPromptTemplate, view_payload


class ScriptedPersonaDriver(PersonaDriver):
    """Picks deterministically from the persona's action space.

    The default, and the one that runs in CI: it needs no API key, adds no
    latency, and makes an episode byte-reproducible end to end. Personas built
    this way are plausible rather than lifelike — they act at human-like
    *times*, but what they do rotates through their permitted actions.

    Its RNG is supplied by the engine, so the same seed yields the same
    choices.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random(0)

    def reset(self, rng: random.Random) -> None:
        # Re-seeded per episode, so rollout N does not continue rollout N-1's
        # random stream — two same-seed episodes must make the same choices.
        self._rng = rng

    def act(self, view: PersonaView) -> PersonaTurn:
        space = view.action_space
        if not space:
            return PersonaTurn(
                persona_id=view.persona.id,
                trigger=view.trigger,
                skipped="no action available",
            )
        spec = space[self._rng.randrange(len(space))]
        params = {
            param.name: self._placeholder(param.type, view)
            for param in spec.params
            if param.required
        }
        return PersonaTurn(
            persona_id=view.persona.id,
            trigger=view.trigger,
            action=Action(type=spec.name, params=params),
            utterance=f"{view.persona.name} performs {spec.name}",
        )

    def _placeholder(self, param_type: str, view: PersonaView) -> object:
        # Integers, never floats: persona actions reach environment state, and
        # the determinism contract rejects floats there.
        if param_type in ("integer", "int", "number"):
            return 0
        if param_type in ("boolean", "bool"):
            return False
        if param_type in ("array", "list"):
            return []
        if param_type in ("object", "dict"):
            return {}
        return f"{view.persona.name} ({view.persona.role or 'colleague'})"


AdapterFactory = Callable[[PersonaPromptTemplate, Task, list], object]


def _default_adapter_factory(agent_id: str) -> AdapterFactory:
    def build(prompt_template, task, tool_specs):
        # Imported lazily: constructing the factory must not require an LLM
        # SDK to be installed for environments that never use one.
        from forge.runtime.agents.factory import make_agent

        return make_agent(
            agent_id,
            prompt_template=prompt_template,
            task=task,
            tool_specs=tool_specs,
        )

    return build


class AgentPersonaDriver(PersonaDriver):
    """Asks a model what this persona would do.

    One adapter is built per persona and reused for the life of the driver, so
    a persona keeps a single, consistent prompt identity rather than being
    reintroduced to the model every turn.

    Failure is contained on purpose. If the adapter raises — no API key, a rate
    limit, a network fault — the turn is skipped and the reason recorded, and
    the episode continues. A simulated colleague who stays quiet is a far
    better failure mode than a rollout that dies because a persona could not
    reach an API.
    """

    def __init__(
        self,
        agent_id: str = "anthropic:claude-sonnet-5",
        adapter_factory: AdapterFactory | None = None,
        fallback: PersonaDriver | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._factory = adapter_factory or _default_adapter_factory(agent_id)
        self._fallback = fallback
        self._adapters: dict[str, object] = {}

    def reset(self, rng: random.Random) -> None:
        # Adapters are dropped between episodes. An adapter that survived would
        # carry one persona's prompt identity into the next rollout, and any
        # provider-side conversation state with it.
        self._adapters.clear()
        if self._fallback is not None:
            self._fallback.reset(rng)

    def act(self, view: PersonaView) -> PersonaTurn:
        if not view.action_space:
            return PersonaTurn(
                persona_id=view.persona.id,
                trigger=view.trigger,
                skipped="no action available",
            )
        try:
            adapter = self._adapter_for(view)
            raw = adapter.act(view_payload(view), view.allowed_action_types)
        except Exception as exc:  # noqa: BLE001 - contained on purpose, see docstring
            if self._fallback is not None:
                return self._fallback.act(view)
            return PersonaTurn(
                persona_id=view.persona.id,
                trigger=view.trigger,
                skipped=f"{type(exc).__name__}: {exc}",
            )

        if not isinstance(raw, dict) or "type" not in raw:
            return PersonaTurn(
                persona_id=view.persona.id,
                trigger=view.trigger,
                skipped=f"driver returned no usable action: {raw!r}",
            )
        utterance = str(raw.pop("_utterance", "")) if isinstance(raw, dict) else ""
        return PersonaTurn(
            persona_id=view.persona.id,
            trigger=view.trigger,
            action=Action.from_dict(raw),
            utterance=utterance,
        )

    def _adapter_for(self, view: PersonaView):
        persona_id = view.persona.id
        adapter = self._adapters.get(persona_id)
        if adapter is None:
            from forge.contracts.persona import PersonaSpec

            spec = PersonaSpec(profile=view.persona, behavior=view.behavior)
            adapter = self._factory(
                PersonaPromptTemplate(spec),
                Task(
                    id=f"persona:{persona_id}",
                    objective=_objective_for(view),
                ),
                list(view.action_space),
            )
            self._adapters[persona_id] = adapter
        return adapter


def _objective_for(view: PersonaView) -> str:
    goals = view.persona.goals
    if goals:
        return "; ".join(goals)
    return f"act as {view.persona.name} would in this situation"


def make_driver(
    driver_id: str,
    rng: random.Random | None = None,
    adapter_factory: AdapterFactory | None = None,
) -> PersonaDriver:
    """Build the driver a population's `driver` field names.

    A model-backed driver always gets the scripted driver as its fallback, so a
    misconfigured API key degrades an environment to plausible personas rather
    than to none.
    """
    if driver_id in ("", "scripted", "none"):
        return ScriptedPersonaDriver(rng)
    return AgentPersonaDriver(
        agent_id=driver_id,
        adapter_factory=adapter_factory,
        fallback=ScriptedPersonaDriver(rng),
    )

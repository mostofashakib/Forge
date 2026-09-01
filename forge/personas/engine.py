"""The runtime that gives simulated humans their turns.

`PersonaEngine` is family-agnostic on purpose. It executes persona actions
through an `ExecutionBackend`, which both the in-process and container families
already expose, so the same cast behaves the same way whether the environment
is a Python state machine or a FastAPI app in a container.

Order of operations for one tick, and each step exists to close a specific
hole:

  1. The scheduler names who is due — deterministic, from the engine's own RNG.
  2. The driver proposes an action for each — possibly a model call.
  3. The guard checks the proposal against that persona's declared space.
  4. Approved actions run through the backend, one at a time, with `ctx.actor_id`
     set to the persona so downstream code can tell a colleague's write from
     the agent's.
  5. Every turn — executed, blocked, or skipped — is recorded.

Step 5 matters as much as step 4. A blocked turn is evidence that a driver left
its action space, and an environment author needs to see that rather than
wonder why a persona went quiet.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

from forge.contracts.persona import (
    PersonaPopulation,
    PersonaSpec,
    PersonaTick,
    PersonaTurn,
    PersonaView,
)
from forge.contracts.types import Action, ToolSpec
from forge.personas.drivers import ScriptedPersonaDriver, make_driver
from forge.personas.guardrails import ActionGuard
from forge.personas.population import population_seed, resolve_roster
from forge.personas.scheduler import PersonaScheduleState, TraitScheduler

if TYPE_CHECKING:
    from forge.contracts.backend import ExecutionBackend
    from forge.contracts.persona import PersonaDriver, PersonaScheduler
    from forge.runtime.context import RuntimeContext

# How many of the step's most recent events a persona is shown. A colleague
# reacts to what just happened, not to the entire history of the shift.
RECENT_EVENT_WINDOW = 8


class PersonaTickResult:
    """What one persona tick did to the world."""

    __slots__ = ("state", "events", "turns")

    def __init__(
        self, state: dict, events: list[dict], turns: list[PersonaTurn]
    ) -> None:
        self.state = state
        self.events = events
        self.turns = turns

    @property
    def acted(self) -> bool:
        return any(turn.executed for turn in self.turns)

    @property
    def blocked(self) -> list[PersonaTurn]:
        return [turn for turn in self.turns if turn.blocked]


class PersonaEngine:
    """Runs a configured cast alongside the agent.

    An engine built from a disabled population is inert: `reset` produces an
    empty roster and `tick` returns the state it was handed, untouched. That is
    deliberate — an environment can construct one unconditionally and pay
    nothing when personas are switched off.
    """

    def __init__(
        self,
        population: PersonaPopulation,
        driver: "PersonaDriver | None" = None,
        scheduler: "PersonaScheduler | None" = None,
        environment_actions: Sequence[str] | None = None,
        tool_specs: Sequence[ToolSpec] | None = None,
    ) -> None:
        self._population = population
        self._schedule_state = PersonaScheduleState()
        self._scheduler = scheduler or TraitScheduler(
            self._schedule_state, max_due=population.max_actions_per_step
        )
        self._guard = ActionGuard(environment_actions, tool_specs)
        self._explicit_driver = driver
        self._driver: "PersonaDriver | None" = driver
        self._roster: list[PersonaSpec] = []
        self._rng = random.Random(0)
        self._transcript: list[PersonaTurn] = []

    # ------------------------------------------------------------------
    # Configuration surface
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._population.enabled

    @property
    def population(self) -> PersonaPopulation:
        return self._population

    @property
    def roster(self) -> list[PersonaSpec]:
        """The cast resolved for the current episode. Empty before `reset`."""
        return list(self._roster)

    @property
    def transcript(self) -> list[PersonaTurn]:
        """Every turn taken this episode, in order — blocked and skipped included."""
        return list(self._transcript)

    def describe(self) -> list[dict]:
        """The cast as the agent should see it: who is here and what they do.

        Traits and knowledge are withheld. An agent that could read a
        colleague's diligence score would optimize against the simulation
        instead of learning the workflow.
        """
        return [
            {
                "id": spec.profile.id,
                "name": spec.profile.name,
                "role": spec.profile.role,
            }
            for spec in self._roster
        ]

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: int) -> list[PersonaSpec]:
        """Resolve the cast for a new episode and clear all per-episode state."""
        self._roster = resolve_roster(self._population, population_seed(self._population, seed))
        self._schedule_state.reset()
        self._transcript = []
        self._rng = random.Random(population_seed(self._population, seed))
        if self._explicit_driver is not None:
            self._driver = self._explicit_driver
        else:
            self._driver = make_driver(self._population.driver, self._rng)
        # Clears whatever the driver carried from the last episode — a seeded
        # RNG position, a per-persona model adapter. Without this, rollout N
        # continues rollout N-1 and two same-seed episodes diverge.
        self._driver.reset(self._rng)
        return self.roster

    @contextmanager
    def deterministic_driver(self):
        """Force the scripted driver for the duration of a block.

        The determinism probe replays an episode and compares hashes. A
        model-backed persona cannot pass that by construction, and it should
        not have to: the probe exists to prove the *environment's* machinery is
        reproducible, not to prove a language model is. Swapping the driver
        keeps the cast present — personas still act, on the same steps, through
        the same guard — while making their choices replayable.
        """
        saved_explicit, saved_driver = self._explicit_driver, self._driver
        probe_driver = ScriptedPersonaDriver(random.Random(0))
        self._explicit_driver = probe_driver
        self._driver = probe_driver
        try:
            yield
        finally:
            self._explicit_driver, self._driver = saved_explicit, saved_driver

    def tick(
        self,
        *,
        backend: "ExecutionBackend",
        state: dict,
        ctx: "RuntimeContext",
        step_index: int,
        agent_action: dict | None = None,
        events: Sequence[dict] | None = None,
    ) -> PersonaTickResult:
        """Give every due persona a turn and apply what survives the guard."""
        step_events = list(events or [])
        if not self._population.enabled or not self._roster:
            return PersonaTickResult(state, [], [])

        tick = PersonaTick(
            step_index=step_index,
            agent_action=agent_action,
            events=step_events,
        )
        due = self._scheduler.due(self._roster, tick, self._rng)

        current_state = state
        produced: list[dict] = []
        turns: list[PersonaTurn] = []
        for persona_id in due:
            spec = self._spec(persona_id)
            if spec is None:
                continue
            # Each persona sees the world its predecessors in this same tick
            # have already changed, so two colleagues acting on one step read
            # as a conversation rather than as simultaneous strangers.
            turn, current_state, new_events = self._take_turn(
                spec, current_state, step_events + produced, step_index, backend, ctx
            )
            turns.append(turn)
            self._transcript.append(turn)
            if turn.executed:
                produced.extend(new_events)
                self._schedule_state.record_action(persona_id, step_index)

        return PersonaTickResult(current_state, produced, turns)

    # ------------------------------------------------------------------

    def _take_turn(
        self,
        spec: PersonaSpec,
        state: dict,
        events: Sequence[dict],
        step_index: int,
        backend: "ExecutionBackend",
        ctx: "RuntimeContext",
    ) -> tuple[PersonaTurn, dict, list[dict]]:
        view = PersonaView(
            persona=spec.profile,
            behavior=spec.behavior,
            step_index=step_index,
            trigger=self._trigger_for(spec.profile.id),
            state=state,
            recent_events=list(events)[-RECENT_EVENT_WINDOW:],
            action_space=self._guard.action_space(spec),
        )
        turn = self._driver.act(view)
        if turn.action is None:
            return (
                turn.model_copy(update={"skipped": turn.skipped or "declined"}),
                state,
                [],
            )

        decision = self._guard.check(spec, turn.action)
        if not decision:
            return turn.model_copy(update={"blocked": decision.reason}), state, []

        try:
            next_state, events = self._execute(
                backend, turn.action, state, ctx, spec, step_index, turn
            )
        except Exception as exc:  # noqa: BLE001
            # A persona whose action the environment rejects is a person who
            # tried something that did not work. That is realistic, and it must
            # not end the agent's episode.
            return (
                turn.model_copy(
                    update={"skipped": f"action failed: {type(exc).__name__}: {exc}"}
                ),
                state,
                [],
            )
        return turn, next_state, events

    def _execute(
        self,
        backend: "ExecutionBackend",
        action: Action,
        state: dict,
        ctx: "RuntimeContext",
        spec: PersonaSpec,
        step_index: int,
        turn: PersonaTurn,
    ) -> tuple[dict, list[dict]]:
        previous_actor = getattr(ctx, "actor_id", "agent")
        ctx.actor_id = spec.profile.id
        try:
            result = backend.execute(action, state, ctx)
        finally:
            ctx.actor_id = previous_actor

        tagged = [
            {**event, "actor": spec.profile.id, "actor_kind": "persona"}
            for event in result.events
        ]
        # Always emit one event for the turn itself, even when the action
        # produced none. Otherwise a persona who acts silently is invisible to
        # the agent, and an agent cannot learn to work with colleagues it
        # cannot observe.
        tagged.append(
            {
                "type": "persona_turn",
                "actor": spec.profile.id,
                "actor_kind": "persona",
                "actor_name": spec.profile.name,
                "actor_role": spec.profile.role,
                "action": action.type,
                "trigger": turn.trigger,
                "step_index": step_index,
                "utterance": turn.utterance,
            }
        )
        return result.state, tagged

    def _trigger_for(self, persona_id: str) -> str:
        getter = getattr(self._scheduler, "trigger_for", None)
        return getter(persona_id) if getter is not None else "scheduled"

    def _spec(self, persona_id: str) -> PersonaSpec | None:
        for spec in self._roster:
            if spec.profile.id == persona_id:
                return spec
        return None

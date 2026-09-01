"""When simulated humans act — the deterministic half of the persona stack.

A scheduler answers only "whose turn is it", never "what do they do". Keeping
those apart is what lets a model choose realistic actions without making the
episode's shape unreproducible: rerun with the same seed and the same personas
are woken on the same steps, whatever they end up saying.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from forge.contracts.persona import (
    TRAIT_MAX,
    PersonaScheduler,
    PersonaSpec,
    PersonaTick,
)


class PersonaScheduleState:
    """Per-episode bookkeeping the scheduler needs and the engine owns.

    Tracks, per persona: the step they were woken on and are waiting out, the
    step they last acted, and how many turns they have spent. Reset between
    episodes so nothing leaks across rollouts.
    """

    def __init__(self) -> None:
        self.woken_at: dict[str, int] = {}
        self.wake_trigger: dict[str, str] = {}
        self.last_acted: dict[str, int] = {}
        self.action_count: dict[str, int] = {}

    def reset(self) -> None:
        self.woken_at.clear()
        self.wake_trigger.clear()
        self.last_acted.clear()
        self.action_count.clear()

    def record_action(self, persona_id: str, step_index: int) -> None:
        self.last_acted[persona_id] = step_index
        self.action_count[persona_id] = self.action_count.get(persona_id, 0) + 1
        self.woken_at.pop(persona_id, None)
        self.wake_trigger.pop(persona_id, None)


class TraitScheduler(PersonaScheduler):
    """Wakes personas from their traits, their triggers, and a seeded RNG.

    Three ways a persona becomes due, checked in this order:

    1. **Woken and ready.** The agent did something in the persona's `wake_on`
       set, and enough steps have passed to cover their reply latency. A
       responsive person replies sooner: latency is scaled down by
       `responsiveness`, so the same configured latency produces a different
       felt delay for an eager colleague than for a slow one.
    2. **Unprompted.** Nobody woke them, but they act on their own initiative,
       at a rate set by `activity` and nudged by the `initiative` trait.
    3. **Neither**, in which case they stay quiet.

    A persona under cooldown, or out of episode budget, is never due — those
    are checked before the three cases above, so a hard budget always wins over
    an eager disposition.
    """

    def __init__(self, state: PersonaScheduleState | None = None, max_due: int = 1) -> None:
        self._state = state or PersonaScheduleState()
        self._max_due = max(1, max_due)

    @property
    def state(self) -> PersonaScheduleState:
        return self._state

    def due(
        self,
        roster: Sequence[PersonaSpec],
        tick: PersonaTick,
        rng: random.Random,
    ) -> list[str]:
        self._note_wakes(roster, tick)

        ready: list[str] = []
        unprompted: list[str] = []
        for spec in roster:
            if not self._eligible(spec, tick.step_index):
                continue
            if self._wake_is_ready(spec, tick.step_index):
                ready.append(spec.profile.id)
            elif self._acts_unprompted(spec, rng):
                unprompted.append(spec.profile.id)

        # Someone the agent is actually waiting on goes before someone who just
        # felt like speaking up, so a busy cast never starves the reply the
        # agent asked for.
        return (ready + unprompted)[: self._max_due]

    def trigger_for(self, persona_id: str) -> str:
        return self._state.wake_trigger.get(persona_id, "unprompted")

    # ------------------------------------------------------------------

    def _note_wakes(self, roster: Sequence[PersonaSpec], tick: PersonaTick) -> None:
        """Record which personas this step's agent action woke."""
        action_type = (tick.agent_action or {}).get("type")
        if action_type is None:
            return
        for spec in roster:
            persona_id = spec.profile.id
            if persona_id in self._state.woken_at:
                continue
            if self._wakes_on(spec, action_type, tick):
                self._state.woken_at[persona_id] = tick.step_index
                self._state.wake_trigger[persona_id] = action_type

    def _wakes_on(self, spec: PersonaSpec, action_type: str, tick: PersonaTick) -> bool:
        wake_on = spec.behavior.wake_on
        if not wake_on:
            # An empty `wake_on` means "anything the agent does concerns me".
            return True
        if action_type in wake_on:
            return True
        # An event addressed to this persona by name wakes them even when the
        # action type itself is not in their list — being spoken to directly
        # outranks a category filter.
        return any(
            event.get("persona_id") == spec.profile.id
            or event.get("recipient") == spec.profile.id
            for event in tick.events
        )

    def _eligible(self, spec: PersonaSpec, step_index: int) -> bool:
        behavior = spec.behavior
        if not behavior.allowed_actions:
            return False
        budget = behavior.max_actions_per_episode
        if budget is not None:
            if self._state.action_count.get(spec.profile.id, 0) >= budget:
                return False
        last = self._state.last_acted.get(spec.profile.id)
        if last is not None and step_index - last <= behavior.cooldown_steps - 1:
            return False
        return True

    def _wake_is_ready(self, spec: PersonaSpec, step_index: int) -> bool:
        woken_at = self._state.woken_at.get(spec.profile.id)
        if woken_at is None:
            return False
        return step_index - woken_at >= self._effective_latency(spec)

    def _effective_latency(self, spec: PersonaSpec) -> int:
        """Configured latency, shortened by responsiveness.

        At responsiveness 0 the configured latency applies in full; at 100 the
        persona replies immediately. Integer arithmetic throughout, so the
        result is stable across platforms.
        """
        latency = spec.behavior.latency_steps
        if latency <= 0:
            return 0
        remaining = TRAIT_MAX - spec.profile.traits.responsiveness
        return (latency * remaining) // TRAIT_MAX

    def _acts_unprompted(self, spec: PersonaSpec, rng: random.Random) -> bool:
        chance = self._unprompted_chance(spec)
        if chance <= 0:
            return False
        return rng.randrange(TRAIT_MAX) < chance

    def _unprompted_chance(self, spec: PersonaSpec) -> int:
        """Blend the configured activity rate with the initiative trait.

        `activity` is the environment author's dial and dominates; `initiative`
        is the persona's own disposition and moves the result by up to half its
        own value. A persona configured inert (activity 0) stays inert however
        eager their disposition — configuration outranks personality.
        """
        activity = spec.behavior.activity
        if activity <= 0:
            return 0
        initiative = spec.profile.traits.initiative
        return min(TRAIT_MAX, activity + (initiative - 50) * activity // (2 * TRAIT_MAX))

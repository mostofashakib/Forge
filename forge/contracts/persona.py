"""Simulated humans that inhabit an environment alongside the agent.

Most Forge environments model a workflow that a real person would not perform
alone: a clinical hand-off, a support queue, an approval chain. An agent that
only ever sees a static world learns to manipulate records; an agent that has
to ask a nurse, wait for a reply, and handle a colleague who answers the wrong
question learns the job.

The split between what is deterministic here and what is not is deliberate,
and it is the whole design:

  * **Who exists and what they are like** is deterministic. A roster is
    resolved from the episode seed, and a persona's traits are fixed integers.
    The same seed always produces the same cast with the same dispositions.
  * **When they act, and how often** is deterministic. `PersonaScheduler`
    draws from a seeded RNG the engine owns — never the environment's — so
    persona timing is reproducible and does not perturb the environment's own
    random stream.
  * **What they actually do** is not. `PersonaDriver` is free to call a model,
    because that is what makes a simulated colleague read like a person rather
    than a state machine.

The third bullet is only safe because of the guardrail: a driver's proposal is
a *request*, and `PersonaBehavior.allowed_actions` bounds what the engine will
execute. A persona that proposes an action outside its declared space has that
turn blocked and recorded, never applied. Everything a persona is permitted to
do is therefore declared up front, in configuration, and shown to the driver as
an explicit action space rather than discovered by trial.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from forge.contracts._arity import check_subclass_arity
from forge.contracts.types import Action, ToolSpec

if TYPE_CHECKING:
    import random


# Traits are integers on a 0-100 scale rather than 0.0-1.0 floats. Persona
# bookkeeping is surfaced in observations and can reach environment state, and
# the determinism contract rejects floats anywhere in state — see
# `_assert_no_floats` in forge/runtime/env_builder.py.
TRAIT_MIN = 0
TRAIT_MAX = 100


class PersonaTraits(BaseModel):
    """The dials that make one simulated human behave unlike another.

    Every trait is deterministic for the life of an episode. They shape *rate*
    and *disposition* — how eagerly someone jumps in, how much they write, how
    careful they are — and they are rendered into the driver's prompt so a
    model-backed persona stays in character rather than drifting toward a
    generic assistant voice.
    """

    responsiveness: int = 50
    """How quickly they act once addressed. Scales down their reply latency."""

    initiative: int = 30
    """How often they act unprompted, with nobody waiting on them."""

    verbosity: int = 50
    """How much they say. Guidance for the driver, not a hard limit."""

    diligence: int = 70
    """How careful they are. Low diligence means omissions and partial answers."""

    formality: int = 50
    """Register, from clipped shorthand to full professional prose."""

    patience: int = 50
    """How long they wait before following up or escalating."""

    @field_validator(
        "responsiveness",
        "initiative",
        "verbosity",
        "diligence",
        "formality",
        "patience",
    )
    @classmethod
    def _within_scale(cls, value: int) -> int:
        if not TRAIT_MIN <= value <= TRAIT_MAX:
            raise ValueError(
                f"trait must be between {TRAIT_MIN} and {TRAIT_MAX}, got {value}"
            )
        return value


class PersonaProfile(BaseModel):
    """Who a persona is. Identity, not behavior.

    `knowledge` is what this person knows that the agent does not — a patient's
    allergy, a policy exception, the reason a ticket was escalated. It reaches
    the driver's prompt and nothing else, so an agent can only obtain it by
    interacting with the persona.
    """

    id: str
    name: str
    role: str = ""
    backstory: str = ""
    goals: list[str] = Field(default_factory=list)
    traits: PersonaTraits = Field(default_factory=PersonaTraits)
    knowledge: dict = Field(default_factory=dict)
    style: str = ""
    """Free-text voice note, e.g. "answers in fragments, never uses greetings"."""


class PersonaBehavior(BaseModel):
    """How a persona engages with the environment.

    This is the guardrail surface. `allowed_actions` is the complete set of
    action types this persona may ever execute; it defaults to empty, so a
    persona configured without one is inert rather than unbounded. The budget
    fields bound how much of an episode the simulated humans can consume.
    """

    allowed_actions: list[str] = Field(default_factory=list)
    """Action types this persona may execute. Empty means the persona is inert."""

    wake_on: list[str] = Field(default_factory=list)
    """Agent action types that make this persona due. Empty means any action."""

    activity: int = 25
    """0-100 chance of acting on a tick where nothing woke them."""

    latency_steps: int = 0
    """Steps between being woken and acting. Reduced by high responsiveness."""

    cooldown_steps: int = 1
    """Steps this persona stays quiet after acting."""

    max_actions_per_episode: int | None = None
    """Hard per-episode budget. None means bounded only by the episode length."""

    @field_validator("activity")
    @classmethod
    def _activity_within_scale(cls, value: int) -> int:
        if not TRAIT_MIN <= value <= TRAIT_MAX:
            raise ValueError(
                f"activity must be between {TRAIT_MIN} and {TRAIT_MAX}, got {value}"
            )
        return value

    @field_validator("latency_steps", "cooldown_steps")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"must not be negative, got {value}")
        return value


class PersonaSpec(BaseModel):
    """One configured persona: who they are, and how they engage."""

    profile: PersonaProfile
    behavior: PersonaBehavior = Field(default_factory=PersonaBehavior)

    @property
    def id(self) -> str:
        return self.profile.id


class PersonaPopulation(BaseModel):
    """The cast configured for an environment.

    `roster` is the explicit cast. `count` is a target size: when it exceeds
    the roster, the remainder is filled by cloning `archetypes` deterministically
    from the episode seed, so "give me eight nurses" does not mean writing eight
    profiles by hand. When `count` is smaller than the roster, the roster is
    truncated deterministically — the cast shrinks, it does not shuffle.
    """

    enabled: bool = False
    count: int | None = None
    roster: list[PersonaSpec] = Field(default_factory=list)
    archetypes: list[PersonaSpec] = Field(default_factory=list)
    max_actions_per_step: int = 1
    """Ceiling on how many personas may act in a single agent step."""

    seed: int | None = None
    """Pins persona scheduling independently of the episode seed when set."""

    driver: str = "scripted"
    """Which driver decides actions: "scripted", or an agent id like
    "anthropic:claude-sonnet-5"."""

    @field_validator("count")
    @classmethod
    def _count_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError(f"count must not be negative, got {value}")
        return value

    @field_validator("max_actions_per_step")
    @classmethod
    def _at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"max_actions_per_step must be >= 1, got {value}")
        return value


class PersonaTick(BaseModel):
    """What the scheduler is allowed to decide on.

    Deliberately narrow: a scheduler sees the step index, what the agent just
    did, and the events it produced. It does not see environment state, because
    a scheduler that reads state is a driver wearing the wrong hat — and it
    would make persona *timing* depend on content the agent controls.
    """

    step_index: int
    agent_action: dict | None = None
    events: list[dict] = Field(default_factory=list)


class PersonaView(BaseModel):
    """Everything a driver sees when deciding one persona's action.

    `action_space` is the persona's permitted actions rendered as tool schemas.
    A driver is expected to choose from it; the engine enforces that it did.
    """

    persona: PersonaProfile
    behavior: PersonaBehavior
    step_index: int
    trigger: str = "scheduled"
    state: dict = Field(default_factory=dict)
    recent_events: list[dict] = Field(default_factory=list)
    action_space: list[ToolSpec] = Field(default_factory=list)

    @property
    def allowed_action_types(self) -> frozenset[str]:
        return frozenset(spec.name for spec in self.action_space)


class PersonaTurn(BaseModel):
    """The outcome of giving one persona a turn.

    Exactly one of `action`, `blocked`, or `skipped` is meaningful. A blocked
    turn is a guardrail rejection and is worth surfacing: it means a driver
    tried to leave its declared action space.
    """

    persona_id: str
    action: Action | None = None
    utterance: str = ""
    trigger: str = "scheduled"
    blocked: str | None = None
    skipped: str | None = None

    @property
    def executed(self) -> bool:
        return self.action is not None and self.blocked is None and self.skipped is None


class PersonaScheduler(ABC):
    """Decides which personas act on a given step, and how often.

    Must be deterministic given `rng`. The engine hands it a dedicated seeded
    RNG rather than the environment's, so persona cadence is reproducible and
    consumes none of the environment's own random stream.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        check_subclass_arity(cls, "due", ("roster", "tick", "rng"))

    @abstractmethod
    def due(
        self,
        roster: Sequence[PersonaSpec],
        tick: PersonaTick,
        rng: "random.Random",
    ) -> list[str]:
        """Persona ids that should take a turn now, in a stable order."""


class PersonaDriver(ABC):
    """Decides what a persona does once the scheduler has given it a turn.

    The one place in the persona stack that may be non-deterministic: an
    implementation is free to call a model. Whatever it proposes is checked
    against the persona's declared action space before it touches the world.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        check_subclass_arity(cls, "act", ("view",))

    @abstractmethod
    def act(self, view: PersonaView) -> PersonaTurn:
        """Choose one action from `view.action_space`, or decline the turn."""

    def reset(self, rng: "random.Random") -> None:
        """Drop everything carried from the previous episode.

        Concrete because a stateless driver has nothing to drop. A driver that
        keeps anything across turns — a seeded RNG position, a model adapter
        holding a persona's conversation — must clear it here. A persona who
        remembers the last rollout is a persona the agent can exhaust across
        episodes, and reward learned against that is not reward the policy can
        reproduce.
        """
        return None

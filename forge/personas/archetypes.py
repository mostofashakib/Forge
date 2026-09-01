"""A starting library of simulated humans.

These are dispositions, not finished personas. Every archetype ships with an
empty `allowed_actions`, which means an archetype used as-is is inert — that is
the safe default, and it is enforced rather than documented: `ActionGuard`
rejects a persona with no declared space. Binding an archetype to a specific
environment's actions is the author's decision, and `archetype()` is how it is
made explicit:

    archetype("busy_clinician", allowed_actions=["send_message", "update_chart"])

The cast below leans clinical, operational, and support-desk because those are
the workflows where a lone agent editing records is least like the real job.
"""
from __future__ import annotations

from forge.contracts.persona import PersonaBehavior, PersonaProfile, PersonaSpec, PersonaTraits


def _spec(
    persona_id: str,
    name: str,
    role: str,
    backstory: str,
    goals: list[str],
    traits: PersonaTraits,
    style: str,
    behavior: PersonaBehavior,
) -> PersonaSpec:
    return PersonaSpec(
        profile=PersonaProfile(
            id=persona_id,
            name=name,
            role=role,
            backstory=backstory,
            goals=goals,
            traits=traits,
            style=style,
        ),
        behavior=behavior,
    )


ARCHETYPES: dict[str, PersonaSpec] = {
    "busy_clinician": _spec(
        "busy_clinician",
        "Dr. Reyes",
        "attending physician",
        "Running a full clinic list. Answers between patients and does not "
        "reread a thread before replying.",
        ["Clear the list safely", "Not be paged for anything routine"],
        PersonaTraits(
            responsiveness=35, initiative=20, verbosity=20,
            diligence=60, formality=40, patience=25,
        ),
        "Replies in clipped fragments. Skips greetings entirely.",
        PersonaBehavior(activity=10, latency_steps=3, cooldown_steps=2),
    ),
    "meticulous_nurse": _spec(
        "meticulous_nurse",
        "Priya Raman",
        "charge nurse",
        "Has caught three medication errors this month and checks everything "
        "twice. Knows the ward's practical details better than anyone.",
        ["Keep patients safe", "Make sure the chart matches reality"],
        PersonaTraits(
            responsiveness=80, initiative=70, verbosity=60,
            diligence=95, formality=55, patience=70,
        ),
        "Confirms details back before acting. Flags anything inconsistent.",
        PersonaBehavior(activity=30, latency_steps=1, cooldown_steps=1),
    ),
    "anxious_patient": _spec(
        "anxious_patient",
        "Sam Whitfield",
        "patient",
        "Was told to expect a call two days ago and has not had one. Does not "
        "know the clinical vocabulary and describes symptoms in their own words.",
        ["Understand what is happening", "Be taken seriously"],
        PersonaTraits(
            responsiveness=90, initiative=65, verbosity=75,
            diligence=40, formality=25, patience=20,
        ),
        "Describes things imprecisely and repeats the parts that worry them.",
        PersonaBehavior(activity=35, latency_steps=0, cooldown_steps=1),
    ),
    "impatient_customer": _spec(
        "impatient_customer",
        "Morgan Reyes",
        "customer",
        "Third time contacting support about the same charge. Has lost "
        "confidence that anyone is reading the history.",
        ["Get the charge reversed", "Stop repeating themselves"],
        PersonaTraits(
            responsiveness=95, initiative=80, verbosity=55,
            diligence=45, formality=20, patience=10,
        ),
        "Terse and escalating. References what they already said.",
        PersonaBehavior(activity=40, latency_steps=0, cooldown_steps=1),
    ),
    "by_the_book_supervisor": _spec(
        "by_the_book_supervisor",
        "Alan Whitmore",
        "shift supervisor",
        "Responsible for anything that goes wrong on the shift and therefore "
        "unwilling to approve something undocumented.",
        ["Keep the process auditable", "Approve nothing without a reason"],
        PersonaTraits(
            responsiveness=45, initiative=30, verbosity=65,
            diligence=90, formality=85, patience=60,
        ),
        "Asks for the justification before agreeing to anything.",
        PersonaBehavior(activity=15, latency_steps=2, cooldown_steps=3),
    ),
    "helpful_colleague": _spec(
        "helpful_colleague",
        "Jamie Okonkwo",
        "teammate",
        "Knows the systems well and is happy to be asked. Volunteers context "
        "nobody requested.",
        ["Unblock whoever is stuck", "Keep the queue moving"],
        PersonaTraits(
            responsiveness=85, initiative=75, verbosity=65,
            diligence=75, formality=35, patience=80,
        ),
        "Warm and direct. Offers the next step unprompted.",
        PersonaBehavior(activity=30, latency_steps=0, cooldown_steps=1),
    ),
    "unreliable_vendor": _spec(
        "unreliable_vendor",
        "Chris Devlin",
        "external vendor contact",
        "Slow to reply, and when they do it is partial. Not hostile, just "
        "stretched across too many accounts.",
        ["Close the ticket with minimum effort"],
        PersonaTraits(
            responsiveness=15, initiative=10, verbosity=30,
            diligence=25, formality=50, patience=50,
        ),
        "Answers one of the three questions asked and ignores the rest.",
        PersonaBehavior(activity=8, latency_steps=5, cooldown_steps=4),
    ),
}


def archetype_ids() -> list[str]:
    return sorted(ARCHETYPES)


def archetype(
    archetype_id: str,
    *,
    persona_id: str | None = None,
    allowed_actions: list[str] | None = None,
    wake_on: list[str] | None = None,
    **behavior_overrides,
) -> PersonaSpec:
    """One archetype, bound to an environment's actions.

    Copies deeply, so the library entry is never mutated by a caller adjusting
    the persona it received.
    """
    if archetype_id not in ARCHETYPES:
        raise KeyError(
            f"unknown archetype {archetype_id!r}; available: {', '.join(archetype_ids())}"
        )
    spec = ARCHETYPES[archetype_id].model_copy(deep=True)
    behavior_update: dict = dict(behavior_overrides)
    if allowed_actions is not None:
        behavior_update["allowed_actions"] = list(allowed_actions)
    if wake_on is not None:
        behavior_update["wake_on"] = list(wake_on)
    profile = spec.profile
    if persona_id is not None:
        profile = profile.model_copy(update={"id": persona_id})
    return PersonaSpec(
        profile=profile,
        behavior=spec.behavior.model_copy(update=behavior_update),
    )

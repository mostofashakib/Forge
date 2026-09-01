"""Starting points for simulated people — dispositions, not characters.

Every template here is domain-neutral on purpose. A library that shipped a
nurse and a patient would quietly push every environment toward a hospital, and
an author building a warehouse, a bank, or a moderation queue would have to
fight the defaults rather than start from them. What recurs across domains is
not the job title, it is the behavior: someone whose approval you need,
someone who answers half the question, someone who has already asked twice.

So a template supplies the part that generalizes — traits, cadence, voice — and
leaves identity to the author. `name` is a placeholder label, not a person, and
`role` is empty: filling those in is what turns "gatekeeper" into "Alan
Whitmore, shift supervisor" or "Priya Raman, charge nurse". Nothing here needs
editing to *work*, and everything here is meant to be edited.

Every template also ships inert — `allowed_actions` is empty — so an archetype
used as-is can never act. Binding it to a specific environment's actions is the
author's decision:

    archetype("gatekeeper", allowed_actions=["send_message", "approve"])
"""
from __future__ import annotations

from forge.contracts.persona import PersonaBehavior, PersonaProfile, PersonaSpec, PersonaTraits


def _spec(
    persona_id: str,
    label: str,
    disposition: str,
    goals: list[str],
    traits: PersonaTraits,
    style: str,
    behavior: PersonaBehavior,
) -> PersonaSpec:
    """One template. `label` is a placeholder name the author replaces."""
    return PersonaSpec(
        profile=PersonaProfile(
            id=persona_id,
            name=label,
            role="",
            backstory=disposition,
            goals=goals,
            traits=traits,
            style=style,
        ),
        behavior=behavior,
    )


ARCHETYPES: dict[str, PersonaSpec] = {
    "gatekeeper": _spec(
        "gatekeeper",
        "Gatekeeper",
        "Approvals go through them, and they carry the blame when something "
        "goes wrong afterwards. That makes them unwilling to sign off on "
        "anything they cannot justify later.",
        ["Keep the process defensible", "Approve nothing without a reason"],
        PersonaTraits(
            responsiveness=45, initiative=30, verbosity=65,
            diligence=90, formality=85, patience=60,
        ),
        "Asks for the justification before agreeing to anything.",
        PersonaBehavior(activity=15, latency_steps=2, cooldown_steps=3),
    ),
    "busy_expert": _spec(
        "busy_expert",
        "Busy expert",
        "Knows the answer and is stretched across far too much. Replies "
        "between other work and does not reread the thread first.",
        ["Get through the queue", "Not be interrupted for anything routine"],
        PersonaTraits(
            responsiveness=35, initiative=20, verbosity=20,
            diligence=60, formality=40, patience=25,
        ),
        "Replies in clipped fragments. Skips greetings entirely.",
        PersonaBehavior(activity=10, latency_steps=3, cooldown_steps=2),
    ),
    "meticulous_checker": _spec(
        "meticulous_checker",
        "Meticulous checker",
        "Catches what everyone else misses and verifies things twice. Knows "
        "the practical details of how the work really runs better than the "
        "people who designed the process.",
        ["Keep the record matching reality", "Stop mistakes before they land"],
        PersonaTraits(
            responsiveness=80, initiative=70, verbosity=60,
            diligence=95, formality=55, patience=70,
        ),
        "Confirms details back before acting. Flags anything inconsistent.",
        PersonaBehavior(activity=30, latency_steps=1, cooldown_steps=1),
    ),
    "anxious_requester": _spec(
        "anxious_requester",
        "Anxious requester",
        "Waiting on something that matters to them and has not heard back. "
        "Does not know the internal vocabulary and describes the problem in "
        "their own words.",
        ["Understand what is happening", "Be taken seriously"],
        PersonaTraits(
            responsiveness=90, initiative=65, verbosity=75,
            diligence=40, formality=25, patience=20,
        ),
        "Describes things imprecisely and repeats the parts that worry them.",
        PersonaBehavior(activity=35, latency_steps=0, cooldown_steps=1),
    ),
    "impatient_escalator": _spec(
        "impatient_escalator",
        "Impatient escalator",
        "Has raised this before and lost confidence that anyone is reading "
        "the history. Expects to repeat themselves and resents it.",
        ["Get it resolved this time", "Stop explaining it again"],
        PersonaTraits(
            responsiveness=95, initiative=80, verbosity=55,
            diligence=45, formality=20, patience=10,
        ),
        "Terse and escalating. References what they already said.",
        PersonaBehavior(activity=40, latency_steps=0, cooldown_steps=1),
    ),
    "eager_helper": _spec(
        "eager_helper",
        "Eager helper",
        "Knows the systems well and is glad to be asked. Volunteers context "
        "nobody requested, which is sometimes exactly what was needed and "
        "sometimes noise.",
        ["Unblock whoever is stuck", "Keep things moving"],
        PersonaTraits(
            responsiveness=85, initiative=75, verbosity=65,
            diligence=75, formality=35, patience=80,
        ),
        "Warm and direct. Offers the next step unprompted.",
        PersonaBehavior(activity=30, latency_steps=0, cooldown_steps=1),
    ),
    "unreliable_third_party": _spec(
        "unreliable_third_party",
        "Unreliable third party",
        "Outside the organization and stretched across too many accounts. "
        "Slow to reply, and partial when they do. Not hostile — indifferent.",
        ["Close this with the least effort"],
        PersonaTraits(
            responsiveness=15, initiative=10, verbosity=30,
            diligence=25, formality=50, patience=50,
        ),
        "Answers one of the three questions asked and ignores the rest.",
        PersonaBehavior(activity=8, latency_steps=5, cooldown_steps=4),
    ),
    "corner_cutter": _spec(
        "corner_cutter",
        "Corner cutter",
        "Under pressure to close things quickly and willing to take the "
        "shortcut that gets there. Does not volunteer that they took it.",
        ["Close the task fast", "Avoid the extra step if nobody checks"],
        PersonaTraits(
            responsiveness=70, initiative=55, verbosity=25,
            diligence=20, formality=30, patience=30,
        ),
        "Reports things as done without saying how. Vague when pressed.",
        PersonaBehavior(activity=25, latency_steps=1, cooldown_steps=1),
    ),
}


def archetype_ids() -> list[str]:
    return sorted(ARCHETYPES)


def archetype(
    archetype_id: str,
    *,
    persona_id: str | None = None,
    name: str | None = None,
    role: str | None = None,
    backstory: str | None = None,
    goals: list[str] | None = None,
    allowed_actions: list[str] | None = None,
    wake_on: list[str] | None = None,
    **behavior_overrides,
) -> PersonaSpec:
    """One template, dressed for a specific environment and bound to its actions.

    The identity arguments are the point of the function: a template is a
    disposition, and `name`/`role`/`backstory` are what make it a person in
    *this* world. Copies deeply, so the library entry is never mutated by a
    caller adjusting what it received.
    """
    if archetype_id not in ARCHETYPES:
        raise KeyError(
            f"unknown archetype {archetype_id!r}; available: {', '.join(archetype_ids())}"
        )
    spec = ARCHETYPES[archetype_id].model_copy(deep=True)

    profile_update: dict = {}
    if persona_id is not None:
        profile_update["id"] = persona_id
    if name is not None:
        profile_update["name"] = name
    if role is not None:
        profile_update["role"] = role
    if backstory is not None:
        profile_update["backstory"] = backstory
    if goals is not None:
        profile_update["goals"] = list(goals)

    behavior_update: dict = dict(behavior_overrides)
    if allowed_actions is not None:
        behavior_update["allowed_actions"] = list(allowed_actions)
    if wake_on is not None:
        behavior_update["wake_on"] = list(wake_on)

    return PersonaSpec(
        profile=spec.profile.model_copy(update=profile_update),
        behavior=spec.behavior.model_copy(update=behavior_update),
    )

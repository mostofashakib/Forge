"""What a model-backed persona reads before it decides.

The prompt has one job the environment's own agent prompt does not: keep a
model in character across a whole episode. Traits are rendered as behavioral
instructions rather than as numbers, because a model handed
``verbosity: 20`` writes a paragraph about being terse, while one told "answer
in a sentence or two, never more" actually does it.

The action space is stated twice — once in prose, once as the tool schemas the
provider enforces — because a persona that wanders outside its space has its
turn blocked by `ActionGuard`, and a blocked turn is a wasted model call.
"""
from __future__ import annotations

import json
from collections.abc import Sequence

from forge.contracts.persona import PersonaSpec, PersonaView
from forge.contracts.types import Observation, Task, ToolSpec
from forge.runtime.prompting import ForgeAgentPromptTemplate


def _band(value: int, low: str, mid: str, high: str) -> str:
    if value <= 33:
        return low
    if value <= 66:
        return mid
    return high


def describe_traits(spec: PersonaSpec) -> list[str]:
    """Trait dials rendered as instructions a model can follow."""
    traits = spec.profile.traits
    return [
        _band(
            traits.verbosity,
            "Keep replies to a sentence or two. Never pad.",
            "Say what is needed and stop.",
            "Explain yourself fully; include the context others may lack.",
        ),
        _band(
            traits.formality,
            "Write casually — fragments, shorthand, no greetings.",
            "Write plainly and professionally.",
            "Write formally, in complete sentences, with proper salutations.",
        ),
        _band(
            traits.diligence,
            "You are stretched thin. You sometimes answer only part of what was "
            "asked, or forget a detail you were told earlier.",
            "You are reliable but not exhaustive.",
            "You are meticulous. You confirm details and flag anything that "
            "looks wrong before acting.",
        ),
        _band(
            traits.patience,
            "You lose patience quickly and say so when something is taking too long.",
            "You are willing to wait a reasonable amount of time.",
            "You are unhurried and never chase anyone.",
        ),
        _band(
            traits.initiative,
            "You do not volunteer information. You answer what you are asked.",
            "You occasionally raise things nobody asked about.",
            "You speak up on your own whenever you notice something relevant.",
        ),
    ]


class PersonaPromptTemplate(ForgeAgentPromptTemplate):
    """Renders one persona's turn.

    Subclasses the standard agent template so tool descriptions stay identical
    to the ones the environment's own agent sees — a persona and an agent
    calling the same action should be reading the same schema.
    """

    def __init__(self, spec: PersonaSpec) -> None:
        super().__init__()
        self._spec = spec

    def system(self, task: Task) -> str:
        profile = self._spec.profile
        lines = [
            f"You are {profile.name}"
            + (f", {profile.role}." if profile.role else "."),
            "",
            "You are a real person inside a simulated workplace, not an assistant. "
            "You are not here to be helpful to an AI agent — you are here to do "
            "your own job. Answer as yourself.",
        ]
        if profile.backstory:
            lines += ["", f"Background: {profile.backstory}"]
        if profile.goals:
            lines += ["", "What you want:"]
            lines += [f"- {goal}" for goal in profile.goals]
        if profile.knowledge:
            lines += [
                "",
                "What you know that others may not. Share it only when it is "
                "relevant, or when someone asks:",
                json.dumps(profile.knowledge, indent=2, sort_keys=True),
            ]
        lines += ["", "How you behave:"]
        lines += [f"- {rule}" for rule in describe_traits(self._spec)]
        if profile.style:
            lines += [f"- {profile.style}"]

        allowed = sorted(set(self._spec.behavior.allowed_actions))
        lines += [
            "",
            "WHAT YOU CAN DO",
            "You act by calling exactly one of the tools below. These are the "
            "only things you are able to do in this environment:",
        ]
        lines += [f"- {name}" for name in allowed] or ["- (nothing)"]
        lines += [
            "",
            "Do not call any other tool, invent an action, or reply in free "
            "text. If nothing on that list is worth doing right now, call no "
            "tool at all — staying quiet is a legitimate choice and is often "
            "the realistic one.",
        ]
        return "\n".join(lines)

    def user(self, observation: Observation, task: Task) -> str:
        payload = json.dumps(observation.payload, sort_keys=True, default=str)
        return (
            f"Situation right now:\n{payload}\n\n"
            "Decide what you — and only you — would do next. Act in character."
        )

    def tool_descriptions(self, tools: Sequence[ToolSpec]) -> list[dict]:
        return super().tool_descriptions(tools)


def view_payload(view: PersonaView) -> dict:
    """The observation dict handed to a persona's model adapter.

    Deliberately not the raw environment state: a persona sees the world as
    *they* would, so this carries who they are, what just happened, and why
    they were given a turn — the context a colleague would have and an
    omniscient observer would not.
    """
    return {
        "you": {
            "id": view.persona.id,
            "name": view.persona.name,
            "role": view.persona.role,
        },
        "step": view.step_index,
        "why_you_are_acting": view.trigger,
        "environment": view.state,
        "recent_events": view.recent_events,
        "actions_available_to_you": sorted(view.allowed_action_types),
    }
